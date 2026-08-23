// SSH transport for the daemon's control traffic over one ControlMaster per
// host. The JSON API uses streamlocal or an exec relay; the separate Herdr
// terminal protocol always uses an exec relay that pane.rs consumes locally.

use std::fs;
use std::os::unix::fs::FileTypeExt;
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use serde::Deserialize;
use tokio::process::Command;
use tokio::time::timeout;

use crate::api::ApiClient;
use crate::config::{ApiTransport, HostConfig};
use crate::util::{err, Logger, Result};

/// Marker in the error text for "the container isn't running". A stopped
/// devcontainer is its resting state, unlike an unreachable ssh host, so the
/// daemon backs off gently instead of treating it as a fault.
pub const DORMANT: &str = "dormant";

/// first build with terminal session observe/control
const MIN_PREVIEW_BUILD: &str = "2026-06-30";

/// Common ssh options, shared by the daemon's master and every pane stream.
pub const SSH_COMMON_OPTS: [&str; 6] = [
    "-o",
    "BatchMode=yes",
    "-o",
    "ServerAliveInterval=15",
    "-o",
    "ServerAliveCountMax=3",
];

#[derive(Debug)]
pub struct RemoteStatus {
    pub socket: String,
    pub supported: bool,
    pub reason: Option<String>,
}

struct SshOutput {
    code: i32,
    out: String,
    err: String,
}

pub(crate) fn ssh_bin() -> std::ffi::OsString {
    #[cfg(test)]
    {
        std::env::var_os("HERDR_MIRROR_TEST_SSH_BIN").unwrap_or_else(|| "ssh".into())
    }
    #[cfg(not(test))]
    {
        "ssh".into()
    }
}

async fn ssh(args: &[String], timeout_ms: u64) -> SshOutput {
    let fut = Command::new(ssh_bin())
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output();
    match timeout(Duration::from_millis(timeout_ms), fut).await {
        Ok(Ok(o)) => SshOutput {
            code: o.status.code().unwrap_or(1),
            out: String::from_utf8_lossy(&o.stdout).into_owned(),
            err: String::from_utf8_lossy(&o.stderr).into_owned(),
        },
        Ok(Err(e)) => SshOutput { code: 1, out: String::new(), err: e.to_string() },
        Err(_) => SshOutput { code: 1, out: String::new(), err: "ssh timeout".into() },
    }
}

fn remove_stale_control_socket(path: &Path) -> Result<()> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => {
            return Err(err(format!(
                "cannot inspect ssh control socket {}: {e}",
                path.display()
            )))
        }
    };
    if !metadata.file_type().is_socket() {
        return Err(err(format!(
            "refusing to replace ssh control path {} because it is not a socket",
            path.display()
        )));
    }
    fs::remove_file(path).map_err(|e| {
        err(format!(
            "cannot remove stale ssh control socket {}: {e}",
            path.display()
        ))
    })
}

pub struct RemoteHost {
    pub cfg: HostConfig,
    ctl_path: PathBuf,
    pub fwd_sock: PathBuf,
    client_sock: PathBuf,
    forwarded: bool,
    /// docker hosts only: resolved container + chosen stdio bridge
    container: Option<crate::docker::Container>,
    /// docker hosts only: owns the relay listener. Dropping it stops serving
    /// and unlinks the socket, so a reconnect never inherits a dead one.
    relay: Option<crate::docker::RelayHandle>,
    /// docker hosts only: companion relay for Herdr's terminal protocol.
    client_relay: Option<crate::docker::RelayHandle>,
    /// ssh hosts only: owns the exec-relay listener when the exec transport
    /// is in use. Same lifecycle reasoning as `relay` above, one level up
    /// the transport stack (ssh exec instead of `docker exec`).
    exec_relay: Option<crate::ssh_relay::RelayHandle>,
    /// ssh hosts only: companion exec relay for Herdr's terminal protocol.
    ssh_client_relay: Option<crate::ssh_relay::RelayHandle>,
    /// ssh hosts only: where the exec relay listens. Deliberately NOT
    /// `fwd_sock`: sharing one path with the streamlocal forward makes the
    /// relay's "is a healthy relay already serving this?" check answerable by
    /// a live `-L` forward, which silently ignores `api_transport = "exec"`
    /// and leaves the daemon believing it is on a relay it never started.
    exec_sock: PathBuf,
    /// ssh hosts only: which transport to try first. Seeded from `cfg` at
    /// construction; `hint_transport` lets the daemon override it with what
    /// last worked, since a fresh `RemoteHost` is built on every reconnect
    /// and would otherwise re-probe streamlocal every time even after it is
    /// known to be dead for this host.
    transport_hint: ApiTransport,
    /// ssh hosts only: which transport this connection actually used, so the
    /// daemon can feed it back into the next `RemoteHost`'s `hint_transport`.
    pub last_api_transport: Option<ApiTransport>,
    log: Logger,
}

/// FNV-1a, truncated to 8 hex chars.
///
/// NOT `DefaultHasher`: this value lands in a path the daemon and every
/// streamer must derive identically, and it has to survive a toolchain
/// upgrade. std's hasher is explicitly unstable across Rust releases, so a
/// bump would silently move every long-named host's socket and orphan its live
/// ControlMaster. FNV-1a is a published algorithm, so the crate's output is
/// fixed by spec rather than by implementation detail. Not a security boundary
/// — it only has to separate a user's own host names, and `socket_stem_hash_is_stable`
/// pins the exact value so a future swap cannot move anyone's sockets unnoticed.
///
/// Hashes the raw bytes rather than going through `Hash for str`, which appends
/// a terminator byte and would give a different (still stable, but arbitrary)
/// value.
fn short_hash(s: &str) -> String {
    use std::hash::Hasher;

    let mut hasher = fnv::FnvHasher::default();
    hasher.write(s.as_bytes());
    format!("{:08x}", hasher.finish() as u32)
}

/// Filename stem shared by a host's sockets, bounded so the longest one
/// still fits sockaddr_un.
///
/// Truncation alone is not enough: two hosts sharing a long prefix (exactly the
/// naming style that overflows the limit in the first place) would collide, and
/// a collision is silent and dangerous rather than loud. `ensure_master` probes
/// the ControlPath before creating a master, so host B would find host A's live
/// master and run every ssh command — including remote-invoke plugin actions —
/// on the wrong machine; the docker relay would unlink the other host's live
/// socket and take over the path. So anything truncated carries a hash of the
/// FULL name.
///
/// Names that already fit are returned verbatim, which is what keeps existing
/// installs on byte-identical paths across this upgrade (no orphaned masters,
/// no migration). Only names that were already too long to work at all move.
fn socket_stem(state_dir: &std::path::Path, host_name: &str) -> String {
    use std::os::unix::ffi::OsStrExt;

    // macOS reserves one byte of sockaddr_un's 104-byte path for NUL; Linux
    // allows 108, and we deliberately apply the tighter bound on both so a
    // hosts.toml is portable. The existing 21-byte overhead covers the longest
    // relay suffix (`-api-client.sock`) and leaves room for OpenSSH's mux temp
    // suffix on the shorter ControlPath.
    const MAX_SOCKET_PATH_BYTES: usize = 103;
    const CONTROL_SOCKET_OVERHEAD: usize = 21;

    let directory_bytes = state_dir.as_os_str().as_bytes().len() + 1;
    let budget = MAX_SOCKET_PATH_BYTES.saturating_sub(directory_bytes + CONTROL_SOCKET_OVERHEAD);
    if host_name.len() <= budget {
        return host_name.into();
    }

    // Degrade in defined steps rather than collapsing to a shared stem: prefix
    // plus hash while both fit, then hash alone. Below that even the hash can't
    // fit, which means the state dir path itself is too long for any socket —
    // return the hash anyway so hosts stay distinct and let the bind fail
    // loudly, rather than handing every host one shared path.
    let hash = short_hash(host_name);
    let keep = budget.saturating_sub(hash.len() + 1);
    let mut end = keep.min(host_name.len());
    while !host_name.is_char_boundary(end) {
        end -= 1;
    }
    let prefix = host_name[..end].trim_end_matches('-');
    if prefix.is_empty() {
        return hash;
    }
    format!("{prefix}-{hash}")
}

/// Match Herdr's API → terminal-client socket derivation exactly: remove one
/// trailing `.sock` when present, then append `-client.sock`. Work bytewise so
/// a non-UTF-8 Unix path stays lossless.
pub(crate) fn client_socket_path(api_socket: &Path) -> PathBuf {
    let raw = api_socket.as_os_str().as_bytes();
    let stem = raw.strip_suffix(b".sock").unwrap_or(raw);
    let mut client = stem.to_vec();
    client.extend_from_slice(b"-client.sock");
    PathBuf::from(std::ffi::OsString::from_vec(client))
}

pub(crate) async fn socket_accepts(path: &Path) -> bool {
    matches!(
        timeout(
            Duration::from_millis(500),
            tokio::net::UnixStream::connect(path)
        )
        .await,
        Ok(Ok(_))
    )
}

pub(crate) fn control_path(state_dir: &std::path::Path, host_name: &str) -> PathBuf {
    state_dir.join(format!("{}.ctl", socket_stem(state_dir, host_name)))
}

pub(crate) fn client_relay_path(state_dir: &std::path::Path, host_name: &str) -> PathBuf {
    state_dir.join(format!("{}-api-client.sock", socket_stem(state_dir, host_name)))
}

impl RemoteHost {
    pub fn new(cfg: &HostConfig, state_dir: &std::path::Path) -> RemoteHost {
        let stem = socket_stem(state_dir, &cfg.name);
        let fwd_sock = state_dir.join(format!("{stem}-api.sock"));
        let exec_sock = state_dir.join(format!("{stem}-api-exec.sock"));
        RemoteHost {
            ctl_path: state_dir.join(format!("{stem}.ctl")),
            client_sock: client_socket_path(&fwd_sock),
            fwd_sock,
            transport_hint: cfg.api_transport,
            cfg: cfg.clone(),
            forwarded: false,
            container: None,
            relay: None,
            client_relay: None,
            exec_relay: None,
            ssh_client_relay: None,
            exec_sock,
            last_api_transport: None,
            log: Logger::new(state_dir, false),
        }
    }

    /// Seed the transport hint from a daemon-remembered choice. A no-op
    /// unless the host is configured `api_transport = "auto"` (the default):
    /// an explicit `socket` or `exec` override always pins its own choice and
    /// ignores anything remembered from a previous connection.
    pub fn hint_transport(&mut self, hint: Option<ApiTransport>) {
        if self.cfg.api_transport == ApiTransport::Auto {
            if let Some(h) = hint {
                self.transport_hint = h;
            }
        }
    }

    /// Bring the transport up: an ssh ControlMaster, or a resolved container.
    ///
    /// ssh hosts take the identical path they always did; the docker branch is
    /// additive.
    pub async fn ensure_ready(&mut self) -> Result<()> {
        if !self.cfg.kind.is_docker() {
            return self.ensure_master().await;
        }
        let bin = self.cfg.docker_bin.clone();
        let ids = crate::docker::resolve(&bin, &self.cfg.kind).await?;
        let Some(id) = ids.first().cloned() else {
            // a stopped devcontainer is the resting state, not a fault; the
            // daemon matches this marker to back off gently
            return Err(err(format!("{DORMANT}: no running container for {}", self.cfg.target)));
        };
        if ids.len() > 1 {
            // reachable without an attacker: a compose devcontainer can put the
            // same local_folder label on several services
            self.log.log(&format!(
                "[{}] {} containers match; using {id} — narrow the config if that is wrong",
                self.cfg.name,
                ids.len()
            ));
        }
        // re-probe on every (re)connect: a rebuilt container may differ
        crate::docker::probe_socat(&bin, &id).await?;
        self.container = Some(crate::docker::Container { id, docker_bin: bin });
        Ok(())
    }

    fn base_args(&self) -> Vec<String> {
        vec![
            "-S".into(),
            self.ctl_path.display().to_string(),
            "-o".into(),
            "BatchMode=yes".into(),
        ]
    }

    fn forward_command(&self, operation: &str, remote_socket: &str) -> Vec<String> {
        let mut args = self.base_args();
        args.extend([
            "-O".into(),
            operation.into(),
            "-L".into(),
            format!("{}:{remote_socket}", self.fwd_sock.display()),
            self.cfg.target.clone(),
        ]);
        args
    }

    pub async fn ensure_master(&mut self) -> Result<()> {
        let mut check = self.base_args();
        check.extend(["-O".into(), "check".into(), self.cfg.target.clone()]);
        if ssh(&check, 15000).await.code == 0 {
            return Ok(());
        }
        self.forwarded = false;
        // OpenSSH falls back to a standalone connection when ControlPath exists
        // but no master is listening. With -f -N that silently leaks one process
        // per retry, while every later -O command keeps targeting the dead socket.
        remove_stale_control_socket(&self.ctl_path)?;
        let mut start: Vec<String> = vec![
            "-M".into(),
            "-S".into(),
            self.ctl_path.display().to_string(),
        ];
        start.extend(SSH_COMMON_OPTS.iter().map(|s| s.to_string()));
        start.extend([
            "-o".into(),
            "ControlPersist=yes".into(),
            "-f".into(),
            "-N".into(),
            self.cfg.target.clone(),
        ]);
        let res = ssh(&start, 20000).await;
        if res.code != 0 {
            return Err(err(format!(
                "ssh master to {} failed: {}",
                self.cfg.target,
                nonempty(&res.err, res.code)
            )));
        }
        let verified = ssh(&check, 15000).await;
        if verified.code != 0 {
            return Err(err(format!(
                "ssh master to {} did not create a usable control socket: {}",
                self.cfg.target,
                nonempty(&verified.err, verified.code)
            )));
        }
        Ok(())
    }

    pub async fn exec(&self, command: &str, timeout_ms: u64) -> Result<String> {
        if let Some(c) = &self.container {
            return c.exec(command, timeout_ms).await;
        }
        let mut args = self.base_args();
        args.extend([self.cfg.target.clone(), command.to_string()]);
        let res = ssh(&args, timeout_ms).await;
        if res.code != 0 {
            return Err(err(format!(
                "ssh exec failed ({command}): {}",
                nonempty(&res.err, res.code)
            )));
        }
        Ok(res.out)
    }

    pub async fn status(&self) -> Result<RemoteStatus> {
        let bin = crate::config::remote_herdr_expr(
            self.cfg.remote_bin.as_deref(),
            self.cfg.session.as_deref(),
        );
        let out = self.exec(&format!("exec {} status --json", bin), 15000).await?;
        #[derive(Deserialize)]
        struct Client {
            version: Option<String>,
        }
        #[derive(Deserialize)]
        struct Server {
            running: Option<bool>,
            socket: Option<String>,
            version: Option<String>,
        }
        #[derive(Deserialize)]
        struct StatusJson {
            client: Option<Client>,
            server: Option<Server>,
        }
        let parsed: StatusJson = serde_json::from_str(&out)?;
        let version = parsed
            .server
            .as_ref()
            .and_then(|s| s.version.clone())
            .or(parsed.client.and_then(|c| c.version))
            .unwrap_or_else(|| "unknown".into());
        let running = parsed.server.as_ref().and_then(|s| s.running) == Some(true);
        let socket = parsed.server.and_then(|s| s.socket).unwrap_or_default();
        let mut status = RemoteStatus { socket, supported: false, reason: None };
        if !running {
            // Name the session, or this reads as "that machine's herdr is
            // down" while the default session is running perfectly and only
            // the configured one is stopped — which is the common way to get
            // here once `session` is in play (a typo, or `herdr session stop`).
            status.reason = Some(match &self.cfg.session {
                Some(name) => format!("remote herdr session {name:?} is not running"),
                None => "remote herdr server is not running".into(),
            });
            return Ok(status);
        }
        match version_supported(&version) {
            Some(true) => status.supported = true,
            Some(false) => {
                status.reason = Some(format!(
                    "remote herdr {version} lacks terminal session streams (need >= 0.7.2 or preview {MIN_PREVIEW_BUILD})"
                ))
            }
            None => status.reason = Some(format!("cannot parse remote version {version}")),
        }
        Ok(status)
    }

    pub async fn forward_api(&mut self, remote_socket: &str) -> Result<PathBuf> {
        if self.forwarded && self.fwd_sock.exists() {
            return Ok(self.fwd_sock.clone());
        }
        // NEVER cancel a healthy forward — other processes may be using it.
        if self.fwd_sock.exists() && ApiClient::connect(&self.fwd_sock).await.is_ok() {
            self.forwarded = true;
            return Ok(self.fwd_sock.clone());
        }
        // A dead process can leave the forward registered on the master with
        // its socket file unlinked — cancel before re-adding.
        let cancel = self.forward_command("cancel", remote_socket);
        let _ = ssh(&cancel, 15000).await;
        let _ = std::fs::remove_file(&self.fwd_sock);
        let forward = self.forward_command("forward", remote_socket);
        let res = ssh(&forward, 15000).await;
        if res.code != 0 {
            return Err(err(format!(
                "ssh socket forward failed: {}",
                nonempty(&res.err, res.code)
            )));
        }
        self.forwarded = true;
        Ok(self.fwd_sock.clone())
    }

    /// Try the streamlocal `-L` forward, verified with a real ping — not just
    /// that `ssh -O forward` reported success.
    ///
    /// The forward registering successfully is not proof the transport works:
    /// some sshds (embedded Go sshds fronting container/VM workspaces are the
    /// case this was written against) accept a direct-streamlocal channel
    /// open and then never service it. Every byte written just sits there, so
    /// the first sign of trouble is the API layer's own connect/ping timing
    /// out or the channel closing with zero bytes read — which is exactly
    /// what `ApiClient::connect`'s ping round-trip surfaces.
    ///
    /// That ping is the client `connect_api` returns, not an extra probe on
    /// top of it: a working host must not pay a round trip for a fallback it
    /// never needs.
    async fn try_socket_transport(&mut self, remote_socket: &str) -> Result<ApiClient> {
        let sock = self.forward_api(remote_socket).await?;
        ApiClient::connect(&sock).await
    }

    /// Drop a forward that just proved itself dead, so it doesn't sit
    /// registered on the ControlMaster for the connection's life with its
    /// socket file unlinked. Unlike `forward_api`'s guard this cannot steal a
    /// healthy forward: it only runs after a real ping failed.
    async fn cancel_forward(&mut self, remote_socket: &str) {
        let args = self.forward_command("cancel", remote_socket);
        let _ = ssh(&args, 15000).await;
        let _ = std::fs::remove_file(&self.fwd_sock);
        self.forwarded = false;
    }

    /// Bridge the remote socket over a plain ssh exec channel instead of a
    /// streamlocal forward. See `ssh_relay` for the transport itself; this
    /// only resolves the relay command once and (re)starts the listener,
    /// mirroring the docker branch below one function down.
    pub(crate) async fn exec_relay_transport(&mut self, remote_socket: &str) -> Result<PathBuf> {
        // NEVER steal a healthy relay — same reasoning as the docker guard:
        // the socket path is per-host but shared across processes (daemon,
        // `remote-*` actions, `once`), and state_dir is a single fixed path.
        // `exec_sock` is the relay's OWN path, so a live streamlocal forward
        // can't answer for it and quietly cancel the exec transport.
        if self.exec_relay.is_none() && ApiClient::connect(&self.exec_sock).await.is_ok() {
            return Ok(self.exec_sock.clone());
        }
        self.exec_relay = None;
        let relay_cmd = crate::ssh_relay::detect_relay_command(self, remote_socket).await?;
        self.log.log(&format!(
            "[{}] exec relay via {} → {remote_socket}",
            self.cfg.name,
            relay_cmd.tool()
        ));
        let handle = crate::ssh_relay::serve_relay(
            self.ctl_path.clone(),
            self.cfg.target.clone(),
            relay_cmd,
            self.exec_sock.clone(),
            self.log.clone(),
        )?;
        let path = handle.path.clone();
        self.exec_relay = Some(handle);
        Ok(path)
    }

    pub(crate) async fn client_relay_transport(
        &mut self,
        remote_socket: &str,
    ) -> Result<PathBuf> {
        if self.ssh_client_relay.as_ref().is_some_and(|relay| relay.path.exists()) {
            return Ok(self.client_sock.clone());
        }
        // A second daemon or a one-shot command may own the stable path. Do
        // not unlink a listener that is accepting connections for live panes.
        if self.ssh_client_relay.is_none() && socket_accepts(&self.client_sock).await {
            return Ok(self.client_sock.clone());
        }
        self.ssh_client_relay = None;
        let remote_client = client_socket_path(Path::new(remote_socket));
        let remote_client = remote_client
            .to_str()
            .ok_or_else(|| err("remote Herdr client socket is not valid UTF-8"))?;
        let relay_cmd = crate::ssh_relay::detect_relay_command(self, remote_client).await?;
        self.log.log(&format!(
            "[{}] terminal client relay via {} → {remote_client}",
            self.cfg.name,
            relay_cmd.tool()
        ));
        let handle = crate::ssh_relay::serve_relay(
            self.ctl_path.clone(),
            self.cfg.target.clone(),
            relay_cmd,
            self.client_sock.clone(),
            self.log.clone(),
        )?;
        let path = handle.path.clone();
        self.ssh_client_relay = Some(handle);
        Ok(path)
    }

    async fn ensure_docker_client_relay(
        &mut self,
        container: &crate::docker::Container,
        remote_socket: &str,
    ) -> Result<PathBuf> {
        if self.client_relay.as_ref().is_some_and(|relay| relay.path.exists()) {
            return Ok(self.client_sock.clone());
        }
        if self.client_relay.is_none() && socket_accepts(&self.client_sock).await {
            return Ok(self.client_sock.clone());
        }
        self.client_relay = None;
        let remote_client = client_socket_path(Path::new(remote_socket));
        let remote_client = remote_client
            .to_str()
            .ok_or_else(|| err("remote Herdr client socket is not valid UTF-8"))?;
        let handle = crate::docker::serve_relay(
            container.clone(),
            remote_client.to_string(),
            self.client_sock.clone(),
            self.log.clone(),
        )?;
        let path = handle.path.clone();
        self.client_relay = Some(handle);
        Ok(path)
    }

    /// Choose and reach the ssh API transport: streamlocal socket forward, or
    /// an exec relay. `api_transport = "socket"` / `"exec"` pin one and never
    /// try the other; the default `"auto"` tries the socket transport first
    /// (unless a prior connection in this daemon's lifetime already learned
    /// it doesn't work here — see `hint_transport`) and falls back to the
    /// exec relay on failure, logging the switch exactly once per fallback.
    ///
    /// Returns a CONNECTED client rather than a path: the connect is the
    /// probe, so the socket transport costs a working host exactly what it
    /// cost before this fallback existed.
    async fn connect_ssh_api(&mut self, remote_socket: &str) -> Result<ApiClient> {
        let configured = self.cfg.api_transport;
        let start_with_socket =
            (if configured == ApiTransport::Auto { self.transport_hint } else { configured })
                != ApiTransport::Exec;

        if start_with_socket {
            match self.try_socket_transport(remote_socket).await {
                Ok(api) => {
                    if let Err(e) = self.client_relay_transport(remote_socket).await {
                        self.log.log(&format!(
                            "[{}] terminal client relay unavailable ({e}) — panes will use direct ssh",
                            self.cfg.name
                        ));
                    }
                    self.last_api_transport = Some(ApiTransport::Socket);
                    return Ok(api);
                }
                // only auto may fall back; an explicit `socket` pin means the
                // caller wants the real failure, not a silent transport swap
                Err(e) if configured != ApiTransport::Auto => return Err(e),
                Err(e) => {
                    self.log.log(&format!(
                        "[{}] streamlocal forward unavailable ({e}) — using exec relay",
                        self.cfg.name
                    ));
                    self.transport_hint = ApiTransport::Exec;
                    // it answered nothing; don't leave it registered
                    self.cancel_forward(remote_socket).await;
                }
            }
        }

        let sock = self.exec_relay_transport(remote_socket).await?;
        let api = ApiClient::connect(&sock).await?;
        if let Err(e) = self.client_relay_transport(remote_socket).await {
            self.log.log(&format!(
                "[{}] terminal client relay unavailable ({e}) — panes will use direct ssh",
                self.cfg.name
            ));
        }
        self.last_api_transport = Some(ApiTransport::Exec);
        Ok(api)
    }

    pub async fn connect_api(&mut self) -> Result<(ApiClient, RemoteStatus)> {
        self.ensure_ready().await?;
        let status = match self.status().await {
            Ok(s) => s,
            Err(_) => {
                // transient mux hiccup (e.g. concurrent -O forward churn) — retry once
                tokio::time::sleep(Duration::from_secs(1)).await;
                self.status().await?
            }
        };
        if !status.supported {
            return Err(err(status.reason.clone().unwrap_or_else(|| "remote unsupported".into())));
        }
        // ssh hosts hand back a connected client (its ping doubles as the
        // transport probe); the docker branch resolves a path and connects below
        let container = self.container.clone();
        let sock = match &container {
            None => return Ok((self.connect_ssh_api(&status.socket).await?, status)),
            Some(c) => {
                // NEVER steal a healthy relay — the socket path is per-HOST but
                // shared across processes (daemon, `remote-*` actions, `once`),
                // and state_dir is deliberately a single fixed path. Binding on
                // top of a live one orphans the owner's listener and then
                // unlinks the path from under it, bouncing the daemon's whole
                // host connection on every remote action. Same reasoning as the
                // ssh forward guard above.
                if self.relay.is_none() && ApiClient::connect(&self.fwd_sock).await.is_ok() {
                    self.fwd_sock.clone()
                } else {
                    self.relay = None;
                    let handle = crate::docker::serve_relay(
                        c.clone(),
                        status.socket.clone(),
                        self.fwd_sock.clone(),
                        self.log.clone(),
                    )?;
                    let path = handle.path.clone();
                    self.relay = Some(handle);
                    path
                }
            }
        };
        let api = ApiClient::connect(&sock).await?;
        if let Err(e) = self
            .ensure_docker_client_relay(container.as_ref().unwrap(), &status.socket)
            .await
        {
            self.log.log(&format!(
                "[{}] terminal client relay unavailable ({e}) — panes will use direct docker",
                self.cfg.name
            ));
        }
        Ok((api, status))
    }

}

fn nonempty(e: &str, code: i32) -> String {
    let t = e.trim();
    if t.is_empty() {
        format!("exit {code}")
    } else {
        t.to_string()
    }
}

/// `Some(true)` = supported, `Some(false)` = too old, `None` = unparseable.
fn version_supported(version: &str) -> Option<bool> {
    let core = version.split(['-', '+']).next()?;
    let mut it = core.split('.');
    let maj: u64 = it.next()?.parse().ok()?;
    let min: u64 = it.next()?.parse().ok()?;
    let pat: u64 = it.next()?.parse().ok()?;
    let newer_than_base = maj > 0 || min > 7 || (min == 7 && pat > 1);
    // preview builds look like 0.7.1-preview.2026-06-30-<hash>
    let preview_ok = version
        .split_once("-preview.")
        .map(|(_, rest)| rest.get(0..10).map(|d| d >= MIN_PREVIEW_BUILD).unwrap_or(false))
        .unwrap_or(false);
    Some(newer_than_base || preview_ok)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_path(name: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "herdr-mirror-{name}-{}-{nonce}",
            std::process::id()
        ))
    }

    fn ssh_host(name: &str) -> HostConfig {
        HostConfig {
            name: name.into(),
            target: format!("{name}.example.com"),
            kind: crate::config::HostKind::Ssh,
            docker_bin: "docker".into(),
            prefix: name.into(),
            remote_bin: None,
            session: None,
            max_cols: None,
            max_rows: None,
            api_transport: ApiTransport::Auto,
            always_control: true,
        }
    }

    #[test]
    fn herdr_client_socket_path_matches_the_real_cli() {
        assert_eq!(
            client_socket_path(Path::new("/tmp/host-api.sock")),
            PathBuf::from("/tmp/host-api-client.sock")
        );
        assert_eq!(
            client_socket_path(Path::new("/tmp/host-api-exec.sock")),
            PathBuf::from("/tmp/host-api-exec-client.sock")
        );
        assert_eq!(
            client_socket_path(Path::new("/tmp/custom")),
            PathBuf::from("/tmp/custom-client.sock")
        );
    }

    #[test]
    fn streamlocal_forward_carries_only_the_api_socket() {
        let state_dir = PathBuf::from("/tmp/herdr-mirror");
        let remote = RemoteHost::new(&ssh_host("work"), &state_dir);

        assert_eq!(
            remote.forward_command("forward", "/run/user/501/herdr.sock"),
            vec![
                "-S",
                "/tmp/herdr-mirror/work.ctl",
                "-o",
                "BatchMode=yes",
                "-O",
                "forward",
                "-L",
                "/tmp/herdr-mirror/work-api.sock:/run/user/501/herdr.sock",
                "work.example.com",
            ]
        );
        assert_eq!(
            client_relay_path(&state_dir, "work"),
            PathBuf::from("/tmp/herdr-mirror/work-api-client.sock")
        );
    }

    #[tokio::test]
    async fn socket_accept_probe_distinguishes_live_and_missing_listeners() {
        let path = test_path("socket-accepts");
        let listener = tokio::net::UnixListener::bind(&path).unwrap();

        assert!(socket_accepts(&path).await);
        drop(listener);
        let _ = fs::remove_file(&path);
        assert!(!socket_accepts(&path).await);
    }

    #[tokio::test]
    async fn docker_client_relay_replaces_stale_socket_and_pumps_bytes() {
        use std::os::unix::fs::PermissionsExt;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let dir = test_path("docker-client-relay");
        fs::create_dir_all(&dir).unwrap();
        let remote_api = dir.join("remote.sock");
        let remote_client = client_socket_path(&remote_api);
        let listener = tokio::net::UnixListener::bind(&remote_client).unwrap();
        let endpoint = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0u8; 4];
            stream.read_exact(&mut request).await.unwrap();
            assert_eq!(&request, b"ping");
            stream.write_all(b"pong").await.unwrap();
        });

        let docker = dir.join("docker");
        fs::write(
            &docker,
            "#!/bin/sh\nremote=${6#UNIX-CONNECT:}\nexec python3 -c 'import socket,sys;s=socket.socket(socket.AF_UNIX);s.connect(sys.argv[1]);s.sendall(sys.stdin.buffer.read(4));sys.stdout.buffer.write(s.recv(4));sys.stdout.buffer.flush()' \"$remote\"\n",
        )
        .unwrap();
        fs::set_permissions(&docker, fs::Permissions::from_mode(0o755)).unwrap();
        let container = crate::docker::Container {
            id: "test".into(),
            docker_bin: docker.display().to_string(),
        };
        let mut cfg = ssh_host("docker");
        cfg.kind = crate::config::HostKind::DockerContainer("test".into());
        let mut remote = RemoteHost::new(&cfg, &dir);
        fs::write(&remote.client_sock, b"stale").unwrap();

        let local = remote
            .ensure_docker_client_relay(&container, remote_api.to_str().unwrap())
            .await
            .unwrap();
        let mode = fs::metadata(&local).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
        let mut client = tokio::net::UnixStream::connect(&local).await.unwrap();
        client.write_all(b"ping").await.unwrap();
        let mut response = [0u8; 4];
        client.read_exact(&mut response).await.unwrap();
        assert_eq!(&response, b"pong");
        endpoint.await.unwrap();

        drop(remote);
        assert!(!local.exists(), "relay owner must unlink its socket on drop");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn long_host_names_use_truncated_socket_paths() {
        let state_dir = PathBuf::from("/Users/example/.local/state/herdr-mirror");
        let name = "remote-development-environment-with-a-very-long-name";
        let remote = RemoteHost::new(&ssh_host(name), &state_dir);
        let stem = socket_stem(&state_dir, name);

        assert!(stem.len() < name.len(), "long name should have been shortened");
        assert!(name.starts_with(stem.split('-').next().unwrap()), "readable prefix kept");
        assert_eq!(
            remote.ctl_path.file_name().unwrap().to_string_lossy(),
            format!("{stem}.ctl")
        );
        // the ControlPath must survive OpenSSH's mux temp suffix on top
        assert!(remote.ctl_path.as_os_str().len() + 17 <= 103);
        assert!(remote.fwd_sock.as_os_str().len() <= 103);
        assert!(remote.client_sock.as_os_str().len() <= 103);
        assert!(remote.exec_sock.as_os_str().len() <= 103);
    }

    #[test]
    fn truncated_stems_never_collide_across_hosts() {
        // The regression this guards: pure truncation gave these one shared
        // stem, so `-O check` found the other host's live master and every ssh
        // command — remote-invoke included — ran on the wrong machine, while
        // the docker relay unlinked the other host's live socket.
        let state_dir = PathBuf::from("/Users/example/.local/state/herdr-mirror");
        let a = "prod-us-east-1-application-server-cluster-node-alpha";
        let b = "prod-us-east-1-application-server-cluster-node-beta";

        // these differ only *past* the cut, so plain truncation collided
        let budget = 103 - state_dir.as_os_str().len() - 1 - 21;
        assert_eq!(a[..budget], b[..budget], "names must be identical up to the cut");

        assert_ne!(
            socket_stem(&state_dir, a),
            socket_stem(&state_dir, b),
            "distinct hosts must not share a socket stem"
        );
        assert_ne!(
            RemoteHost::new(&ssh_host(a), &state_dir).ctl_path,
            RemoteHost::new(&ssh_host(b), &state_dir).ctl_path
        );
    }

    #[test]
    fn socket_stem_hash_is_stable() {
        // Golden values. These bytes are baked into live socket paths, so a
        // change here silently relocates every long-named host's ControlMaster.
        // If a hashing swap ever moves them, this must fail first.
        assert_eq!(short_hash("vps"), "4f02d738");
        assert_eq!(short_hash("prod-us-east-1-application-server-cluster-node-alpha"), "cbd62d65");
    }

    #[test]
    fn short_host_names_keep_their_exact_path() {
        // existing installs must not move to a new socket on upgrade, which
        // would orphan a live ControlMaster
        let state_dir = PathBuf::from("/Users/example/.local/state/herdr-mirror");
        assert_eq!(socket_stem(&state_dir, "vps"), "vps");
        assert_eq!(socket_stem(&state_dir, "work"), "work");
        assert_eq!(
            control_path(&state_dir, "vps"),
            state_dir.join("vps.ctl"),
            "path derivation must match what pre-upgrade daemons used"
        );
    }

    #[test]
    fn socket_stem_survives_multibyte_names_and_a_tiny_budget() {
        let state_dir = PathBuf::from("/Users/example/.local/state/herdr-mirror");
        // truncation must land on a char boundary, never split a code point
        let name = "höst-nàme-with-ünicode-and-a-very-long-tail-that-overflows";
        let stem = socket_stem(&state_dir, name);
        assert!(stem.is_char_boundary(stem.len()));

        // a state dir long enough to eat the whole budget still yields distinct
        // stems rather than one shared path
        let deep = PathBuf::from("/Users/example/".to_string() + &"d".repeat(80));
        assert_ne!(socket_stem(&deep, "alpha-host-name"), socket_stem(&deep, "beta-host-name"));
        assert!(!socket_stem(&deep, "alpha-host-name").is_empty());
    }

    #[test]
    fn removes_stale_control_socket() {
        let path = test_path("stale-control");
        let listener = std::os::unix::net::UnixListener::bind(&path).unwrap();
        drop(listener);

        remove_stale_control_socket(&path).unwrap();

        assert!(!path.exists());
    }

    #[test]
    fn refuses_to_replace_non_socket_control_path() {
        let path = test_path("non-socket-control");
        fs::write(&path, "do not delete").unwrap();

        let error = remove_stale_control_socket(&path).unwrap_err().to_string();

        assert!(error.contains("is not a socket"));
        assert_eq!(fs::read_to_string(&path).unwrap(), "do not delete");
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn version_gate() {
        assert_eq!(version_supported("0.7.1"), Some(false));
        assert_eq!(version_supported("0.7.2"), Some(true));
        assert_eq!(version_supported("0.8.0"), Some(true));
        assert_eq!(version_supported("1.0.0"), Some(true));
        assert_eq!(version_supported("0.7.1-preview.2026-06-30-3459798b606d"), Some(true));
        assert_eq!(version_supported("0.7.1-preview.2026-07-04-aaaa"), Some(true));
        assert_eq!(version_supported("0.7.1-preview.2026-06-29-aaaa"), Some(false));
        assert_eq!(version_supported("garbage"), None);
    }
}

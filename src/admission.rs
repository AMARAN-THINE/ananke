//! Admission control and cooperative cancellation for CPU-expensive endpoints.
//!
//! The problem this solves: dropping an axum future does NOT kill a
//! `spawn_blocking` task. A client that fires a carrier-route request and hangs
//! up leaves the solve running to its full budget on a core of the Deck. Repeat
//! that and the box is saturated by work nobody is waiting for.
//!
//! Two halves:
//!
//! * The `OwnedSemaphorePermit` from a `HeavySlot` is moved INTO the blocking
//!   task, so a slot stays claimed for the real duration of the CPU work rather
//!   than the lifetime of the connection.
//! * `CancelOnDrop` lives in the async handler. When the connection dies the
//!   future drops, the guard drops, and a flag flips. `Deadline::should_stop()`
//!   polls that flag alongside the wall-clock budget, so every solver loop bails
//!   at its next iteration instead of running to completion.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Instant;

use axum::{
    extract::{Request, State},
    http::{header, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Cheap to clone. Injected into handlers via `Extension<Admission>`.
#[derive(Clone)]
pub struct Admission {
    heavy: Arc<Semaphore>,
    light: Arc<Semaphore>,
}

/// A claimed heavy slot. Split it before spawning the solve.
pub struct HeavySlot {
    permit: OwnedSemaphorePermit,
    cancel: Arc<AtomicBool>,
}

impl HeavySlot {
    /// Returns `(permit, cancel_flag)`.
    ///
    /// Move `permit` into the `spawn_blocking` closure. Keep a `CancelOnDrop`
    /// built from `cancel_flag` alive in the async handler, and hand clones of
    /// the flag to the `Deadline`s the solver polls.
    pub fn split(self) -> (OwnedSemaphorePermit, Arc<AtomicBool>) {
        (self.permit, self.cancel)
    }
}

/// Flips a cancellation flag when dropped. Hold one in the async handler for
/// the lifetime of the request.
pub struct CancelOnDrop(Arc<AtomicBool>);

impl CancelOnDrop {
    pub fn new(flag: Arc<AtomicBool>) -> Self {
        Self(flag)
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

/// A wall-clock budget plus a cancellation flag. Solver loops poll
/// `should_stop()` where they previously compared `Instant::elapsed()` against a
/// constant.
#[derive(Clone)]
pub struct Deadline {
    start: Instant,
    budget_ms: u128,
    flag: Arc<AtomicBool>,
}

impl Deadline {
    /// Starts the clock now.
    pub fn new(budget_ms: u128, flag: Arc<AtomicBool>) -> Self {
        Self {
            start: Instant::now(),
            budget_ms,
            flag,
        }
    }

    /// Wall-clock budget exhausted.
    pub fn expired(&self) -> bool {
        self.start.elapsed().as_millis() > self.budget_ms
    }

    /// Client went away.
    pub fn cancelled(&self) -> bool {
        self.flag.load(Ordering::Relaxed)
    }

    /// Either. This is the one solver loops should call.
    pub fn should_stop(&self) -> bool {
        self.cancelled() || self.expired()
    }
}

impl Admission {
    /// `heavy`: max concurrent route solves. On the Deck (4c/8t) use 2.
    /// `light`: max concurrent everything-else requests.
    pub fn new(heavy: usize, light: usize) -> Self {
        Self {
            heavy: Arc::new(Semaphore::new(heavy)),
            light: Arc::new(Semaphore::new(light)),
        }
    }

    /// Claim a heavy slot, or `None` if the server is at capacity.
    pub fn try_heavy(&self) -> Option<HeavySlot> {
        self.heavy
            .clone()
            .try_acquire_owned()
            .ok()
            .map(|permit| HeavySlot {
                permit,
                cancel: Arc::new(AtomicBool::new(false)),
            })
    }

    #[allow(dead_code)]
    pub fn heavy_free(&self) -> usize {
        self.heavy.available_permits()
    }

    #[allow(dead_code)]
    pub fn light_free(&self) -> usize {
        self.light.available_permits()
    }
}

/// 503 with a Retry-After hint.
pub fn overloaded(retry_after_secs: u32) -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        [(header::RETRY_AFTER, retry_after_secs.to_string())],
        "server at capacity, retry shortly\n",
    )
        .into_response()
}

/// Tuple form for handlers whose error type is `(StatusCode, String)`.
pub fn overloaded_tuple() -> (StatusCode, String) {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        "Server at capacity — all route-solve slots are busy. Retry shortly.".to_string(),
    )
}

/// Coarse guard for the heavy router subtree.
///
/// NOT used any more: the heavy handlers claim their own `HeavySlot` so the
/// permit can travel into `spawn_blocking`. Running both would draw twice on the
/// same semaphore and halve effective concurrency. Kept for reference only.
#[allow(dead_code)]
pub async fn heavy_guard(State(adm): State<Admission>, req: Request, next: Next) -> Response {
    let Ok(permit) = adm.heavy.clone().try_acquire_owned() else {
        return overloaded(30);
    };
    let resp = next.run(req).await;
    drop(permit);
    resp
}

/// Guard for everything else. Stops a scrape flood from starving the runtime.
pub async fn light_guard(State(adm): State<Admission>, req: Request, next: Next) -> Response {
    let Ok(permit) = adm.light.clone().try_acquire_owned() else {
        return overloaded(5);
    };
    let resp = next.run(req).await;
    drop(permit);
    resp
}

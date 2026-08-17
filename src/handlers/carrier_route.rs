use axum::{extract::State, http::StatusCode, Extension, Json};
use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap, HashSet},
    sync::Arc,
    time::Instant,
};
use tracing::{info, warn};

use crate::admission::{overloaded_tuple, Admission, CancelOnDrop, Deadline};
use crate::config::{CARRIER_JUMP_RANGE, CARRIER_REFINE_BUDGET_MS};
use crate::models::CarrierRouteQuery;
use crate::state::AppState;

// ── Greedy phase tuning ──────────────────────────────────────────────────────

/// Alternative next-hops retained per stack frame. Backtracking can only escape
/// a cul-de-sac by trying a *different* hop, so a frame holding one candidate is
/// a frame with no escape route. 32 is far more than any real void needs and
/// keeps per-frame memory negligible.
const GREEDY_CANDIDATES_PER_HOP: usize = 32;

/// Hard ceiling on DFS iterations (pushes + pops + rescans), so a pathological
/// region can't spin forever.
const GREEDY_MAX_ATTEMPTS: usize = 250_000;

/// Wall-clock ceiling for the whole greedy phase. Atlas gives greedy-engine
/// carrier requests 180s, so this has to land comfortably inside that.
const GREEDY_BUDGET_MS: u128 = 60_000;

/// How far the path is allowed to exceed the straight-line jump count before the
/// DFS gives up on the current branch and backtracks instead of wandering.
const GREEDY_HOP_SLACK: f64 = 6.0;
const GREEDY_MIN_HOPS: usize = 200;

/// Ceiling on the A* corridor preload. Past this the node maps alone run to
/// gigabytes, which OOMs a Deck-class box; clearing the set makes A* find
/// nothing and the caller falls back to the greedy path.
const ASTAR_MAX_PRELOAD: usize = 1_500_000;

// ── Tritium fuel model ───────────────────────────────────────────────────────
//
//   fuel = round(BASE_FUEL_PER_JUMP
//                + distance * (capacityUsed + fuelInReservoir + carrierMass)
//                  / FUEL_MASS_DIVISOR)
//
// capacityUsed is the carrier's used capacity: crew, cargo, reserved cargo
// space, ship packs and module packs. fuelInReservoir is the jump tank, capped
// at 1,000 t. Squadron carriers are LIGHTER than fleet carriers and therefore
// cheaper to jump, which the old 60,000 t constant had backwards.
//
// Verified against a squadron carrier at 3,920 t used capacity with a full
// reservoir over 500 Ly: round(5 + 500 * 19,920 / 200,000) = 55 t, matching
// observed in-game consumption. Note the in-game UI frequently disagrees with
// what the jump actually costs; the journal is the ground truth, not the
// navigator panel.

/// Flat cost applied to every jump, including 0 Ly in-system jumps.
const BASE_FUEL_PER_JUMP: f64 = 5.0;
/// Divisor in the fuel formula.
const FUEL_MASS_DIVISOR: f64 = 200_000.0;
/// Hull mass term for a standard Drake-class Fleet Carrier.
const FLEET_CARRIER_MASS: f64 = 25_000.0;
/// Hull mass term for a Squadron Carrier.
const SQUADRON_CARRIER_MASS: f64 = 15_000.0;
/// Jump reservoir capacity. The tank tops up from stored tritium to this, not
/// to whatever the carrier happened to be carrying when the route was plotted.
const CARRIER_RESERVOIR_CAPACITY: f64 = 1_000.0;

#[derive(Clone)]
struct Cand {
    id: i64,
    name: String,
    x: f64,
    y: f64,
    z: f64,
    d_dst: f64,
}

/// One node on the DFS stack. `stage` tracks how hard we've already looked from
/// here: 0 = not yet scanned, 1 = scanned along the destination bearing,
/// 2 = scanned the full jump-range sphere. A frame is only dead once stage 2 is
/// exhausted.
struct Frame {
    id: i64,
    name: String,
    x: f64,
    y: f64,
    z: f64,
    cands: Vec<Cand>,
    cursor: usize,
    stage: u8,
}

enum GreedyAct {
    ScanBearing,
    ScanSphere,
    Advance(Cand),
    Backtrack,
}

/// Scan one axis-aligned box, keeping the `GREEDY_CANDIDATES_PER_HOP` systems
/// closest to the destination that are actually inside carrier jump range of
/// `from` and not already on the stack or proven dead.
///
/// Ranking purely by distance-to-destination is deliberate: every hop that makes
/// progress sorts ahead of every lateral or backward hop for free, so there is no
/// separate "must improve" filter left to dead-end on. The DFS prefers progress
/// and only takes a sideways hop when nothing better remains.
fn scan_box(
    tx: &rusqlite::Transaction<'_>,
    bbox: (f64, f64, f64, f64, f64, f64),
    from: (f64, f64, f64),
    dest: (f64, f64, f64),
    seen: &HashSet<i64>,
    out: &mut Vec<Cand>,
) -> Result<(), String> {
    let (mn_x, mx_x, mn_y, mx_y, mn_z, mx_z) = bbox;

    let mut stmt = tx
        .prepare_cached(
            "
        SELECT s.id64, s.name, i.minX, i.minY, i.minZ
        FROM systems_index i JOIN systems s ON i.id = s.id64
        WHERE i.minX BETWEEN ? AND ? AND i.minY BETWEEN ? AND ? AND i.minZ BETWEEN ? AND ?
    ",
        )
        .map_err(|e| e.to_string())?;

    let rows = stmt
        .query_map(
            rusqlite::params![mn_x, mx_x, mn_y, mx_y, mn_z, mx_z],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, f64>(2)?,
                    row.get::<_, f64>(3)?,
                    row.get::<_, f64>(4)?,
                ))
            },
        )
        .map_err(|e| e.to_string())?;

    let rsq = CARRIER_JUMP_RANGE * CARRIER_JUMP_RANGE;

    for row in rows.filter_map(Result::ok) {
        let (id, name, nx, ny, nz) = row;
        if seen.contains(&id) {
            continue;
        }

        let d2 = (nx - from.0).powi(2) + (ny - from.1).powi(2) + (nz - from.2).powi(2);
        if d2 > rsq || d2 <= 0.0 {
            continue;
        }

        let d_dst = ((nx - dest.0).powi(2) + (ny - dest.1).powi(2) + (nz - dest.2).powi(2)).sqrt();

        if out.len() == GREEDY_CANDIDATES_PER_HOP && d_dst >= out[out.len() - 1].d_dst {
            continue;
        }
        let pos = out.partition_point(|c| c.d_dst < d_dst);
        out.insert(
            pos,
            Cand {
                id,
                name,
                x: nx,
                y: ny,
                z: nz,
                d_dst,
            },
        );
        out.truncate(GREEDY_CANDIDATES_PER_HOP);
    }

    Ok(())
}

/// Tier 1: tight boxes around the ideal hop point, 498.5 LY along the bearing to
/// the destination, widening 20 -> 140 LY. Cheap, and in normal-density space it
/// lands a near-optimal hop on the first or second try.
fn collect_near_bearing(
    tx: &rusqlite::Transaction<'_>,
    from: (f64, f64, f64),
    dest: (f64, f64, f64),
    seen: &HashSet<i64>,
) -> Result<Vec<Cand>, String> {
    let mut out: Vec<Cand> = Vec::with_capacity(GREEDY_CANDIDATES_PER_HOP);

    let v = (dest.0 - from.0, dest.1 - from.1, dest.2 - from.2);
    let v_mag = (v.0 * v.0 + v.1 * v.1 + v.2 * v.2).sqrt();
    if v_mag <= 1e-9 {
        return Ok(out);
    }

    let (ux, uy, uz) = (v.0 / v_mag, v.1 / v_mag, v.2 / v_mag);
    let reach = 498.5f64.min(v_mag);
    let (tx_, ty_, tz_) = (
        from.0 + ux * reach,
        from.1 + uy * reach,
        from.2 + uz * reach,
    );

    let mut r = 20.0f64;
    for _ in 0..5 {
        scan_box(
            tx,
            (tx_ - r, tx_ + r, ty_ - r, ty_ + r, tz_ - r, tz_ + r),
            from,
            dest,
            seen,
            &mut out,
        )?;
        if !out.is_empty() {
            break;
        }
        r += 30.0;
    }
    Ok(out)
}

/// Tier 2: the entire jump-range sphere around the current system.
///
/// This is the tier the original code never had. It only ever looked in a
/// +/-140 LY box around the ideal hop point, so a sparse patch dead ahead killed
/// the whole route even when thousands of perfectly usable systems sat off-axis
/// or closer in. That is the "Could not find a star within range" failure. It is
/// the expensive tier near the core, which is exactly why it's the fallback and
/// not the default.
fn collect_full_sphere(
    tx: &rusqlite::Transaction<'_>,
    from: (f64, f64, f64),
    dest: (f64, f64, f64),
    seen: &HashSet<i64>,
) -> Result<Vec<Cand>, String> {
    let mut out: Vec<Cand> = Vec::with_capacity(GREEDY_CANDIDATES_PER_HOP);
    let r = CARRIER_JUMP_RANGE;
    scan_box(
        tx,
        (
            from.0 - r,
            from.0 + r,
            from.1 - r,
            from.1 + r,
            from.2 - r,
            from.2 + r,
        ),
        from,
        dest,
        seen,
        &mut out,
    )?;
    Ok(out)
}

async fn do_carrier_route(
    state: Arc<AppState>,
    admission: Admission,
    params: CarrierRouteQuery,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    // Claim a heavy slot up front. The permit is moved into the blocking task
    // below, so the slot stays held for the real duration of the solve rather
    // than the lifetime of the connection.
    let slot = admission.try_heavy().ok_or_else(overloaded_tuple)?;
    let (permit, cancel) = slot.split();

    // Dropped when this future is dropped, i.e. when the client disconnects.
    // That flips the flag every solver loop below polls.
    let _disconnect_guard = CancelOnDrop::new(cancel.clone());

    let pool = state.db_pool.clone();
    let vk = state.vulkan_astar.clone();

    let result = tokio::task::spawn_blocking(move || -> Result<serde_json::Value, String> {
        let _permit = permit;
        let mut conn = pool.get().map_err(|e| e.to_string())?;
        let t_start = Instant::now();
        let greedy_deadline = Deadline::new(GREEDY_BUDGET_MS, cancel.clone());

        #[derive(Clone)]
        struct CNode { g: u32, f: f64, id: i64 }
        impl PartialEq for CNode { fn eq(&self, o: &Self) -> bool { self.id == o.id } }
        impl Eq for CNode {}
        impl PartialOrd for CNode { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
        impl Ord for CNode {
            fn cmp(&self, o: &Self) -> Ordering { o.f.partial_cmp(&self.f).unwrap_or(Ordering::Equal) }
        }

        let (src_id, src_name, x1, y1, z1) = crate::procgen::resolve_system(&conn, &params.current_system)?;
        let (dest_id, dest_name, x2, y2, z2) = crate::procgen::resolve_system(&conn, &params.destination)?;

        let total_distance = ((x2 - x1).powi(2) + (y2 - y1).powi(2) + (z2 - z1).powi(2)).sqrt();
        let base_cargo = params.used_cargo;
        let is_squadron = params.is_squadron.unwrap_or(false);
        let carrier_base_mass = if is_squadron { SQUADRON_CARRIER_MASS } else { FLEET_CARRIER_MASS };
        // The reservoir refills to its own capacity between jumps, not to the
        // level it happened to be at when the route was requested. Taking the
        // max keeps a caller who reports more than 1,000 t from being clamped
        // downward mid-simulation.
        let max_tank_capacity = CARRIER_RESERVOIR_CAPACITY.max(params.tank_fuel);

        info!("Carrier route: {} -> {}, {:.0} LY, {} carrier",
            params.current_system, params.destination, total_distance,
            if is_squadron { "squadron" } else { "personal" });

        // ── Greedy baseline ───────────────────────────────────────────────────
        //
        // Depth-first greedy with backtracking. The old version was a straight
        // walk: pick the single best next hop, and if the candidate box came
        // back empty, abort the entire route. That has two failure modes, and
        // Beagle Point -> Sagittarius A* hits both. It only searched a small box
        // around the ideal 498.5 LY hop point, so it never saw systems that were
        // off-axis or closer in; and it required each hop to strictly reduce
        // distance-to-destination, so the first cul-de-sac was fatal. 500 LY is
        // enormous and density climbs toward the core, so an empty candidate set
        // out there means over-filtering, not empty space.
        //
        // Now: each stack frame keeps its best N alternatives, escalates from a
        // bearing scan to a full sphere scan before giving up, and on exhaustion
        // pops back to its parent and tries the next alternative there. A node
        // popped off the stack stays in `seen`, so it is never re-entered — its
        // whole subtree is already known to lead nowhere regardless of how it
        // was reached.
        let mut greedy_path: Vec<(i64, String, f64, f64, f64)> = Vec::new();
        {
            let tx = conn.transaction().map_err(|e| e.to_string())?;

            let mut stack: Vec<Frame> = vec![Frame {
                id: src_id, name: src_name.clone(),
                x: x1, y: y1, z: z1,
                cands: Vec::new(), cursor: 0, stage: 0,
            }];

            let mut seen: HashSet<i64> = HashSet::new();
            seen.insert(src_id);

            let max_hops = (((total_distance / CARRIER_JUMP_RANGE) * GREEDY_HOP_SLACK).ceil()
                as usize).max(GREEDY_MIN_HOPS);

            let mut attempts = 0usize;
            let mut deepest_hops = 0usize;
            let mut deepest_name = src_name.clone();

            loop {
                attempts += 1;
                if attempts > GREEDY_MAX_ATTEMPTS {
                    return Err(format!(
                        "Route failed: gave up searching for a carrier-range path to '{}' after {} attempts. Deepest point reached was '{}' at {} jumps.",
                        dest_name, GREEDY_MAX_ATTEMPTS, deepest_name, deepest_hops));
                }
                if greedy_deadline.cancelled() {
                    return Err("Route cancelled: client disconnected.".to_string());
                }
                if greedy_deadline.expired() {
                    return Err(format!(
                        "Route failed: greedy search exceeded {}s looking for a carrier-range path to '{}'. Deepest point reached was '{}' at {} jumps.",
                        GREEDY_BUDGET_MS / 1000, dest_name, deepest_name, deepest_hops));
                }

                let (hx, hy, hz) = {
                    let f = stack.last().unwrap();
                    (f.x, f.y, f.z)
                };

                let d_rem = ((x2 - hx).powi(2) + (y2 - hy).powi(2) + (z2 - hz).powi(2)).sqrt();
                if d_rem <= CARRIER_JUMP_RANGE { break; }

                if stack.len() - 1 > deepest_hops {
                    deepest_hops = stack.len() - 1;
                    deepest_name = stack.last().unwrap().name.clone();
                }

                let act = {
                    let f = stack.last().unwrap();
                    if stack.len() >= max_hops      { GreedyAct::Backtrack }
                    else if f.stage == 0            { GreedyAct::ScanBearing }
                    else if f.cursor < f.cands.len(){ GreedyAct::Advance(f.cands[f.cursor].clone()) }
                    else if f.stage == 1            { GreedyAct::ScanSphere }
                    else                            { GreedyAct::Backtrack }
                };

                match act {
                    GreedyAct::ScanBearing => {
                        let c = collect_near_bearing(&tx, (hx, hy, hz), (x2, y2, z2), &seen)?;
                        let f = stack.last_mut().unwrap();
                        f.cands = c; f.cursor = 0; f.stage = 1;
                    }
                    GreedyAct::ScanSphere => {
                        let c = collect_full_sphere(&tx, (hx, hy, hz), (x2, y2, z2), &seen)?;
                        let f = stack.last_mut().unwrap();
                        f.cands = c; f.cursor = 0; f.stage = 2;
                    }
                    GreedyAct::Advance(c) => {
                        stack.last_mut().unwrap().cursor += 1;
                        // The list was built before deeper hops were taken, so
                        // re-check rather than trusting it.
                        if seen.contains(&c.id) { continue; }
                        seen.insert(c.id);
                        stack.push(Frame {
                            id: c.id, name: c.name,
                            x: c.x, y: c.y, z: c.z,
                            cands: Vec::new(), cursor: 0, stage: 0,
                        });
                    }
                    GreedyAct::Backtrack => {
                        if stack.len() == 1 {
                            return Err(format!(
                                "Route failed: no carrier-range path exists from '{}' to '{}' in the known galaxy. Deepest point reached was '{}' at {} jumps.",
                                src_name, dest_name, deepest_name, deepest_hops));
                        }
                        stack.pop();
                    }
                }
            }

            for f in &stack {
                greedy_path.push((f.id, f.name.clone(), f.x, f.y, f.z));
            }
            greedy_path.push((dest_id, dest_name.clone(), x2, y2, z2));
        }

        let greedy_jumps = greedy_path.len() - 1;
        info!("Carrier greedy: {} jumps in {}ms", greedy_jumps, t_start.elapsed().as_millis());

        let use_astar = params.engine.as_deref().unwrap_or("greedy").to_lowercase() != "greedy";
        let mut astar_path: Option<Vec<(i64, String, f64, f64, f64)>> = None;

        // ── A* refinement ─────────────────────────────────────────────────────
        if use_astar {
            // Corridor half-width was a flat 1500 LY regardless of route length.
            // On a short bubble hop that padded the preload AABB to ~3100 LY per
            // side, dragging most of the inhabited galaxy into node_pos/node_name
            // before A* ran a single expansion — which is how a 100 LY route
            // managed to blow a 120s budget. Scale it instead, with a floor
            // comfortably above one jump so short routes keep room to deviate.
            let corridor_half = (total_distance * 0.25).clamp(600.0, 1500.0);
            let corridor_sq   = corridor_half * corridor_half;
            let dv            = (x2 - x1, y2 - y1, z2 - z1);
            let dv_len_sq     = dv.0*dv.0 + dv.1*dv.1 + dv.2*dv.2;

            // The deadline covers the preload, not just the search. The scan
            // below is the single longest uninterruptible stretch in this
            // handler, so leaving it outside the deadline made a disconnected
            // client cost a full corridor scan before anything noticed.
            let t_astar_start = Instant::now();
            let astar_deadline = Deadline::new(CARRIER_REFINE_BUDGET_MS, cancel.clone());

            // No join to `systems` and no name column. Names are needed for the
            // ~24-80 nodes on the final path, not for the up to 1.5M scanned
            // here, and allocating a String per row was the bulk of the cost of
            // a scan that gets thrown away whenever the cap trips.
            let mut preload_stmt = conn.prepare("
                SELECT i.id, i.minX, i.minY, i.minZ
                FROM systems_index i
                WHERE i.minX BETWEEN ? AND ? AND i.minY BETWEEN ? AND ? AND i.minZ BETWEEN ? AND ?
            ").map_err(|e| e.to_string())?;

            let buf = corridor_half;
            let mut all_systems: Vec<(i64, f64, f64, f64)> = Vec::new();
            let mut corridor_overflow = false;
            let mut preload_aborted = false;
            {
                let rows = preload_stmt.query_map(
                    rusqlite::params![x1.min(x2)-buf, x1.max(x2)+buf, y1.min(y2)-buf, y1.max(y2)+buf, z1.min(z2)-buf, z1.max(z2)+buf],
                    |r| Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?, r.get::<_, f64>(2)?, r.get::<_, f64>(3)?))
                ).map_err(|e| e.to_string())?;

                let mut scanned: u64 = 0;
                for row in rows {
                    // Cheap enough to poll often, rare enough not to matter.
                    scanned += 1;
                    if scanned % 65_536 == 0 && astar_deadline.should_stop() {
                        preload_aborted = true;
                        break;
                    }

                    let Ok((id, nx, ny, nz)) = row else { continue };

                    if dv_len_sq >= 1.0 {
                        let w = (nx - x1, ny - y1, nz - z1);
                        let t = ((w.0*dv.0 + w.1*dv.1 + w.2*dv.2) / dv_len_sq).clamp(0.0, 1.0);
                        let (px, py, pz) = (x1 + t*dv.0, y1 + t*dv.1, z1 + t*dv.2);
                        if (nx-px).powi(2) + (ny-py).powi(2) + (nz-pz).powi(2) > corridor_sq { continue; }
                    }

                    all_systems.push((id, nx, ny, nz));
                    if all_systems.len() > ASTAR_MAX_PRELOAD {
                        corridor_overflow = true;
                        break;
                    }
                }
            }

            // Over the cap the node maps alone run to gigabytes. Dropping the set
            // leaves A* with nothing to expand, so it returns None and the caller
            // falls back to the greedy path — slower route, but the box survives.
            if corridor_overflow {
                warn!("Carrier A* corridor exceeded {} systems after {}ms — falling back to greedy ({} jumps)",
                    ASTAR_MAX_PRELOAD, t_astar_start.elapsed().as_millis(), greedy_jumps);
                all_systems.clear();
            } else if preload_aborted {
                warn!("Carrier A* corridor preload abandoned after {}ms (cancelled or out of budget) — falling back to greedy ({} jumps)",
                    t_astar_start.elapsed().as_millis(), greedy_jumps);
                all_systems.clear();
            }

            info!("Carrier A* corridor preload: {} systems (half-width {:.0} LY, {}ms)",
                all_systems.len(), corridor_half, t_astar_start.elapsed().as_millis());

            // Build CPU node map (needed for path→response conversion regardless of GPU/CPU path)
            let mut node_pos: HashMap<i64, (f64, f64, f64)> = HashMap::with_capacity(all_systems.len() + 2);
            let cell_size = (CARRIER_JUMP_RANGE * 0.9).max(50.0);

            for &(id, nx, ny, nz) in &all_systems {
                node_pos.insert(id, (nx, ny, nz));
            }
            node_pos.entry(src_id).or_insert((x1, y1, z1));
            node_pos.entry(dest_id).or_insert((x2, y2, z2));
            let h_fn = |x: f64, y: f64, z: f64| -> f64 {
                (((x-x2).powi(2)+(y-y2).powi(2)+(z-z2).powi(2)).sqrt() / CARRIER_JUMP_RANGE).ceil()
            };

            // Helper: convert id64 path to the tuple vec the rest of the function
            // expects. Names are left empty and filled in once, for the selected
            // path only, after A* has picked a winner.
            let ids_to_path = |ids: Vec<i64>| -> Vec<(i64, String, f64, f64, f64)> {
                ids.iter().map(|&id| {
                    let (nx, ny, nz) = node_pos.get(&id).copied().unwrap_or((0., 0., 0.));
                    (id, String::new(), nx, ny, nz)
                }).collect()
            };

            // ── Try GPU A* ────────────────────────────────────────────────────
            let gpu_ids: Option<Vec<i64>> = (|| {
                let vk = vk.as_ref()?;
                let nodes_f32: Vec<(i64, f32, f32, f32)> = node_pos.iter()
                    .map(|(&id, &(nx, ny, nz))| (id, nx as f32, ny as f32, nz as f32))
                    .collect();
                let graph = vk.build_graph(&nodes_f32, cell_size as f32)?;
                if total_distance > 5_000.0 {
                    vk.run_bidirectional(
                        &graph,
                        src_id, dest_id,
                        src_id, dest_id, 0, 0,   // no bridge seeding needed for carrier
                        CARRIER_JUMP_RANGE as f32,
                        greedy_jumps as u32,
                        &astar_deadline,
                    )
                } else {
                    vk.run_unidirectional(
                        &graph,
                        src_id, dest_id,
                        CARRIER_JUMP_RANGE as f32,
                        greedy_jumps as u32,
                        &astar_deadline,
                    )
                }
            })();

            if let Some(ids) = gpu_ids {
                info!("Carrier GPU A* found {} jumps (greedy {}), {}ms",
                    ids.len() - 1, greedy_jumps, t_astar_start.elapsed().as_millis());
                astar_path = Some(ids_to_path(ids));
            } else {
                // ── CPU A* fallback ───────────────────────────────────────────
                if vk.is_some() {
                    info!("GPU A* returned no improvement, running CPU fallback");
                }

                // Build spatial grid (only needed for CPU path)
                let mut grid: HashMap<(i32, i32, i32), Vec<i64>> = HashMap::new();
                for (&id, &(nx, ny, nz)) in &node_pos {
                    let cell = ((nx/cell_size) as i32, (ny/cell_size) as i32, (nz/cell_size) as i32);
                    grid.entry(cell).or_default().push(id);
                }

                let cpu_ids: Option<Vec<i64>> = if total_distance > 5_000.0 {
                    let h_bwd = |x: f64, y: f64, z: f64| -> f64 {
                        (((x-x1).powi(2)+(y-y1).powi(2)+(z-z1).powi(2)).sqrt() / CARRIER_JUMP_RANGE).ceil()
                    };
                    (|| {
                        let mut fwd_cf:     HashMap<i64, i64> = HashMap::new();
                        let mut bwd_cf:     HashMap<i64, i64> = HashMap::new();
                        let mut fwd_g:      HashMap<i64, u32> = HashMap::new();
                        let mut bwd_g:      HashMap<i64, u32> = HashMap::new();
                        let mut fwd_closed: HashSet<i64>      = HashSet::new();
                        let mut bwd_closed: HashSet<i64>      = HashSet::new();
                        let mut fwd_open:   BinaryHeap<CNode> = BinaryHeap::new();
                        let mut bwd_open:   BinaryHeap<CNode> = BinaryHeap::new();

                        fwd_g.insert(src_id, 0);
                        bwd_g.insert(dest_id, 0);
                        fwd_open.push(CNode { g: 0, f: h_fn(x1, y1, z1), id: src_id });
                        bwd_open.push(CNode { g: 0, f: h_bwd(x2, y2, z2), id: dest_id });

                        let mut mu: u32          = greedy_jumps as u32;
                        let mut best_meeting: Option<i64> = None;
                        let rsq = CARRIER_JUMP_RANGE * CARRIER_JUMP_RANGE;

                        macro_rules! expand_carrier {
                            ($cx:expr, $cy:expr, $cz:expr, $id:expr,
                             $my_g:expr, $my_g_map:expr, $my_cf:expr, $my_open:expr,
                             $my_closed:expr, $other_g_map:expr, $h_fn_local:expr) => {{
                                let (bx, by, bz) = (($cx/cell_size) as i32, ($cy/cell_size) as i32, ($cz/cell_size) as i32);
                                for dx in -2i32..=2 { for dy in -2i32..=2 { for dz in -2i32..=2 {
                                    if let Some(v) = grid.get(&(bx+dx, by+dy, bz+dz)) {
                                        for &n_id in v {
                                            if n_id == $id || $my_closed.contains(&n_id) { continue; }
                                            let (nx, ny, nz) = match node_pos.get(&n_id) { Some(&p) => p, None => continue };
                                            let d2 = (nx-$cx).powi(2)+(ny-$cy).powi(2)+(nz-$cz).powi(2);
                                            if d2 > rsq || d2 == 0.0 { continue; }
                                            let tg = $my_g + 1;
                                            if tg < *$my_g_map.get(&n_id).unwrap_or(&u32::MAX) {
                                                $my_g_map.insert(n_id, tg);
                                                $my_cf.insert(n_id, $id);
                                                $my_open.push(CNode { g: tg, f: tg as f64 + $h_fn_local(nx, ny, nz), id: n_id });
                                                if let Some(&og) = $other_g_map.get(&n_id) {
                                                    let total = tg + og;
                                                    if total < mu { mu = total; best_meeting = Some(n_id); }
                                                }
                                            }
                                        }
                                    }
                                }}}
                            }};
                        }

                        loop {
                            if astar_deadline.should_stop() { break; }
                            if fwd_open.is_empty() && bwd_open.is_empty() { break; }
                            let fwd_min_g = fwd_open.peek().map(|n| n.g).unwrap_or(u32::MAX);
                            let bwd_min_g = bwd_open.peek().map(|n| n.g).unwrap_or(u32::MAX);
                            if fwd_min_g.saturating_add(bwd_min_g) >= mu { break; }

                            let fwd_min_f = fwd_open.peek().map(|n| n.f).unwrap_or(f64::MAX);
                            let bwd_min_f = bwd_open.peek().map(|n| n.f).unwrap_or(f64::MAX);
                            let expand_fwd = fwd_min_f <= bwd_min_f;

                            if expand_fwd {
                                let Some(CNode { g, id, .. }) = fwd_open.pop() else { continue; };
                                if g >= mu || fwd_closed.contains(&id) { continue; }
                                fwd_closed.insert(id);
                                if let Some(&bg) = bwd_g.get(&id) {
                                    let total = g + bg;
                                    if total < mu { mu = total; best_meeting = Some(id); }
                                }
                                let (cx, cy, cz) = match node_pos.get(&id) { Some(&p) => p, None => continue };
                                let d_dst = ((cx-x2).powi(2)+(cy-y2).powi(2)+(cz-z2).powi(2)).sqrt();
                                if d_dst <= CARRIER_JUMP_RANGE {
                                    let tg = g + 1;
                                    if tg < *fwd_g.get(&dest_id).unwrap_or(&u32::MAX) {
                                        fwd_g.insert(dest_id, tg); fwd_cf.insert(dest_id, id);
                                        fwd_open.push(CNode { g: tg, f: tg as f64, id: dest_id });
                                        if tg < mu { mu = tg; best_meeting = Some(dest_id); }
                                    }
                                }
                                expand_carrier!(cx, cy, cz, id, g, fwd_g, fwd_cf, fwd_open, fwd_closed, bwd_g, h_fn);
                            } else {
                                let Some(CNode { g, id, .. }) = bwd_open.pop() else { continue; };
                                if g >= mu || bwd_closed.contains(&id) { continue; }
                                bwd_closed.insert(id);
                                if let Some(&fg) = fwd_g.get(&id) {
                                    let total = fg + g;
                                    if total < mu { mu = total; best_meeting = Some(id); }
                                }
                                let (cx, cy, cz) = match node_pos.get(&id) { Some(&p) => p, None => continue };
                                let d_src = ((cx-x1).powi(2)+(cy-y1).powi(2)+(cz-z1).powi(2)).sqrt();
                                if d_src <= CARRIER_JUMP_RANGE {
                                    let tg = g + 1;
                                    if tg < *bwd_g.get(&src_id).unwrap_or(&u32::MAX) {
                                        bwd_g.insert(src_id, tg); bwd_cf.insert(src_id, id);
                                        bwd_open.push(CNode { g: tg, f: tg as f64, id: src_id });
                                        if tg < mu { mu = tg; best_meeting = Some(src_id); }
                                    }
                                }
                                expand_carrier!(cx, cy, cz, id, g, bwd_g, bwd_cf, bwd_open, bwd_closed, fwd_g, h_bwd);
                            }
                        }

                        let m = best_meeting?;
                        let mut fwd_path: Vec<i64> = vec![m];
                        let mut cur = m;
                        while cur != src_id { match fwd_cf.get(&cur) { Some(&p) => { cur = p; fwd_path.push(cur); } None => return None, } }
                        fwd_path.reverse();
                        let mut bwd_path: Vec<i64> = Vec::new();
                        let mut cur = m;
                        while cur != dest_id { match bwd_cf.get(&cur) { Some(&p) => { cur = p; bwd_path.push(cur); } None => break, } }
                        if *bwd_path.last().unwrap_or(&m) != dest_id { bwd_path.push(dest_id); }
                        fwd_path.extend(bwd_path);
                        Some(fwd_path)
                    })()
                } else {
                    // Unidirectional
                    (|| {
                        let mut came_from: HashMap<i64, i64> = HashMap::new();
                        let mut g_score:   HashMap<i64, u32> = HashMap::new();
                        let mut closed:    HashSet<i64>      = HashSet::new();
                        let mut open:      BinaryHeap<CNode> = BinaryHeap::new();
                        let rsq = CARRIER_JUMP_RANGE * CARRIER_JUMP_RANGE;

                        g_score.insert(src_id, 0);
                        open.push(CNode { g: 0, f: h_fn(x1, y1, z1), id: src_id });

                        while let Some(CNode { g, id, .. }) = open.pop() {
                            if astar_deadline.should_stop() { return None; }
                            if g as usize >= greedy_jumps { continue; }
                            if id == dest_id {
                                let mut path = vec![dest_id];
                                let mut cur = dest_id;
                                while cur != src_id { match came_from.get(&cur) { Some(&p) => { cur = p; path.push(cur); } None => return None, } }
                                path.reverse();
                                return Some(path);
                            }
                            if closed.contains(&id) { continue; }
                            closed.insert(id);
                            let (cx, cy, cz) = match node_pos.get(&id) { Some(&p) => p, None => continue };
                            let d_dst = ((cx-x2).powi(2)+(cy-y2).powi(2)+(cz-z2).powi(2)).sqrt();
                            if d_dst <= CARRIER_JUMP_RANGE && !closed.contains(&dest_id) {
                                let tg = g + 1;
                                if tg < *g_score.get(&dest_id).unwrap_or(&u32::MAX) {
                                    g_score.insert(dest_id, tg); came_from.insert(dest_id, id);
                                    open.push(CNode { g: tg, f: tg as f64, id: dest_id });
                                }
                            }
                            let (bx, by, bz) = ((cx/cell_size) as i32, (cy/cell_size) as i32, (cz/cell_size) as i32);
                            for dx in -2i32..=2 { for dy in -2i32..=2 { for dz in -2i32..=2 {
                                if let Some(v) = grid.get(&(bx+dx, by+dy, bz+dz)) {
                                    for &n_id in v {
                                        if n_id == id || closed.contains(&n_id) { continue; }
                                        let (nx, ny, nz) = match node_pos.get(&n_id) { Some(&p) => p, None => continue };
                                        let d2 = (nx-cx).powi(2)+(ny-cy).powi(2)+(nz-cz).powi(2);
                                        if d2 > rsq || d2 == 0.0 { continue; }
                                        let tg = g + 1;
                                        if tg < *g_score.get(&n_id).unwrap_or(&u32::MAX) {
                                            g_score.insert(n_id, tg); came_from.insert(n_id, id);
                                            open.push(CNode { g: tg, f: tg as f64 + h_fn(nx, ny, nz), id: n_id });
                                        }
                                    }
                                }
                            }}}
                        }
                        None
                    })()
                };

                if let Some(ids) = cpu_ids {
                    info!("Carrier CPU A* found {} jumps (greedy {}), {}ms",
                        ids.len() - 1, greedy_jumps, t_astar_start.elapsed().as_millis());
                    astar_path = Some(ids_to_path(ids));
                } else {
                    info!("Carrier A* did not improve on greedy ({} jumps), {}ms",
                        greedy_jumps, t_start.elapsed().as_millis());
                }
            }
        } else {
            info!("Engine: greedy only, skipping A* refinement ({} jumps)", greedy_jumps);
        }

        // ── Select final path ─────────────────────────────────────────────────
        let mut final_path = if use_astar { astar_path.unwrap_or(greedy_path) } else { greedy_path };
        let is_optimal = final_path.len() - 1 < greedy_jumps;

        // The greedy path already carries names from its own scan; an A* path
        // does not, because the preload no longer fetches them. Resolve only
        // what ended up on the route.
        if final_path.iter().any(|(_, name, _, _, _)| name.is_empty()) {
            let ids: Vec<i64> = final_path.iter().map(|p| p.0).collect();
            let names = crate::handlers::neutron_route::batch_resolve_names(&conn, &ids)?;
            for entry in final_path.iter_mut() {
                if entry.1.is_empty() {
                    entry.1 = match entry.0 {
                        id if id == src_id  => src_name.clone(),
                        id if id == dest_id => dest_name.clone(),
                        id => names.get(&id).cloned().unwrap_or_default(),
                    };
                }
            }
        }

        // ── Fuel simulation ───────────────────────────────────────────────────
        let mut tank = params.tank_fuel;
        let mut market = params.stored_tritium;
        let mut total_fuel_used = 0.0;
        let mut jumps_json: Vec<serde_json::Value> = Vec::with_capacity(final_path.len());

        for step in 0..final_path.len() {
            let (nid, ref nname, nx, ny, nz) = final_path[step];
            let dist_from_start = ((nx - x1).powi(2) + (ny - y1).powi(2) + (nz - z1).powi(2)).sqrt();
            let dist_to_dest    = ((nx - x2).powi(2) + (ny - y2).powi(2) + (nz - z2).powi(2)).sqrt();

            if step == 0 {
                jumps_json.push(serde_json::json!({
                    "system": nname, "id64": nid.to_string(),
                    "distance_from_start": 0.0,
                    "distance_to_destination": (dist_to_dest * 100.0).round() / 100.0,
                    "jump_distance": 0.0, "fuel_used": 0.0,
                    "fuel_left_tank": tank, "tritium_in_market": market,
                    "has_enough_fuel": true
                }));
            } else {
                let (_, _, px, py, pz) = final_path[step - 1];
                let jdist = ((nx - px).powi(2) + (ny - py).powi(2) + (nz - pz).powi(2)).sqrt();
                let c = base_cargo + market;
                let r = tank.max(0.0);
                let jump_fuel = (BASE_FUEL_PER_JUMP
                    + (jdist * (c + r + carrier_base_mass)) / FUEL_MASS_DIVISOR)
                    .round()
                    .max(BASE_FUEL_PER_JUMP);
                total_fuel_used += jump_fuel;
                tank -= jump_fuel;
                let has_enough_fuel = tank >= 0.0;
                let top_off = (max_tank_capacity - tank).max(0.0).min(market);
                tank += top_off;
                market -= top_off;
                jumps_json.push(serde_json::json!({
                    "system": nname, "id64": nid.to_string(),
                    "distance_from_start": (dist_from_start * 100.0).round() / 100.0,
                    "distance_to_destination": (dist_to_dest * 100.0).round() / 100.0,
                    "jump_distance": (jdist * 100.0).round() / 100.0,
                    "fuel_used": jump_fuel, "fuel_left_tank": tank,
                    "tritium_in_market": market, "has_enough_fuel": has_enough_fuel
                }));
            }
        }

        Ok(serde_json::json!({
            "source": params.current_system, "destination": params.destination,
            "is_squadron": is_squadron, "optimised": is_optimal,
            "total_distance_ly": (total_distance * 100.0).round() / 100.0,
            "base_cargo_capacity_used": base_cargo,
            "initial_fuel_tank": params.tank_fuel,
            "initial_market_tritium": params.stored_tritium,
            "total_fuel_used": total_fuel_used,
            "final_fuel_tank": tank, "final_market_tritium": market,
            "totalJumps": jumps_json.len().saturating_sub(1),
            "route": jumps_json
        }))
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Route task failed: {}", e)))?
    .map_err(|e| (StatusCode::BAD_REQUEST, e))?;

    Ok(Json(result))
}

pub async fn carrier_route_post(
    State(state): State<Arc<AppState>>,
    Extension(admission): Extension<Admission>,
    Json(params): Json<CarrierRouteQuery>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    do_carrier_route(state, admission, params).await
}

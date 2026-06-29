use axum::{extract::State, http::StatusCode, Json};
use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap, HashSet},
    sync::Arc,
    time::Instant,
};
use tracing::info;

use crate::config::NEUTRON_REFINE_BUDGET_MS as REFINE_BUDGET_MS;
use crate::models::NeutronRouteQuery;
use crate::state::AppState;

// ═══════════════════════════════════════════════════════════════════════════
// Persistent NeutronGraph — resident nodes + spatial grid only.
//
// Previous versions stored a CSR adjacency list capped at MAX_DEGREE neighbours
// per node. That cap was added to stop the galactic core from exploding edge
// memory past 9GB, but it conflated two different things:
//
//   * connectivity  — does *a* path exist?
//   * optimality     — is it the *fewest-jump* path?
//
// In dense regions the MAX_DEGREE nearest neutrons are all short-range, so the
// stored graph held no edge long enough to actually use a 6x / ~450 LY boosted
// jump. Every boosted hop silently collapsed to the local Nth-nearest distance,
// so the router always returned more jumps than Spansh. Distance-stratifying the
// kept edges helped but still plateaued: ANY fixed sparsification can cut the
// edge the optimal path needs.
//
// This version stores NO edges. It keeps the coordinate arrays and the coarse
// spatial grid, and generates neighbours ON-DEMAND within the EXACT boosted
// range of the current request. The search therefore runs over the true
// reachability graph, so an admissible-heuristic A* returns a provably minimal
// jump count. Memory is bounded because edges are transient per expansion and
// only ever materialised for the handful of nodes the search actually visits —
// and the ~512MB of resident CSR edge arrays are gone entirely, which also makes
// startup near-instant (no edge precompute, which was the slow / OOM-prone part).
// ═══════════════════════════════════════════════════════════════════════════

/// Grid cell size in LY. A ±2 cell window (125 cells) covers a ~450 LY boosted
/// range with margin; larger ranges simply widen the window. Sized once at
/// startup and shared across all requests regardless of their jump range.
const GRID_CELL_LY: f32 = 250.0;

#[allow(dead_code)]
pub struct NeutronGraph {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub z: Vec<f32>,
    pub id64: Vec<i64>,
    pub id_map: HashMap<i64, u32>,

    // Coarse spatial grid for O(1) on-demand neighbour generation.
    pub grid_cell_size: f32,
    pub grid_cells: HashMap<(i32, i32, i32), (u32, u32)>,
    pub grid_nodes: Vec<u32>,

    // Retained for the Vulkan GPU path, which is currently bypassed in routing
    // (see do_neutron_route). Kept populated so it can be re-enabled without a
    // rebuild once the GPU kernel is verified to produce exact hop counts.
    pub vk_nodes: Vec<(i64, f32, f32, f32)>,
}

#[allow(dead_code)]
impl NeutronGraph {
    /// Builds the resident graph from the SQLite database. Call this ONCE at
    /// startup. No edge precompute: just node coordinates + the spatial grid.
    pub fn build(conn: &rusqlite::Connection) -> Result<Self, String> {
        info!("Building resident NeutronGraph (nodes + spatial grid)...");
        let t_start = Instant::now();

        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut z = Vec::new();
        let mut id64 = Vec::new();
        let mut id_map = HashMap::new();
        let mut vk_nodes = Vec::new();

        let mut stmt = conn
            .prepare(
                "SELECT ns.systemId64, i.minX, i.minY, i.minZ \
                 FROM neutron_systems ns \
                 JOIN systems_index i ON ns.systemId64 = i.id",
            )
            .map_err(|e| e.to_string())?;

        let rows = stmt
            .query_map([], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, f64>(1)? as f32,
                    r.get::<_, f64>(2)? as f32,
                    r.get::<_, f64>(3)? as f32,
                ))
            })
            .map_err(|e| e.to_string())?;

        for row in rows {
            if let Ok((id, nx, ny, nz)) = row {
                let idx = x.len() as u32;
                x.push(nx);
                y.push(ny);
                z.push(nz);
                id64.push(id);
                id_map.insert(id, idx);
                vk_nodes.push((id, nx, ny, nz));
            }
        }
        let n = x.len();
        info!("Loaded {} neutrons. Building spatial grid...", n);

        let cell_size = GRID_CELL_LY;
        let mut buckets: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::new();
        for i in 0..n {
            let cell = (
                (x[i] / cell_size).floor() as i32,
                (y[i] / cell_size).floor() as i32,
                (z[i] / cell_size).floor() as i32,
            );
            buckets.entry(cell).or_default().push(i as u32);
        }

        let mut grid_nodes = Vec::with_capacity(n);
        let mut grid_cells = HashMap::with_capacity(buckets.len());
        for (key, indices) in buckets {
            let offset = grid_nodes.len() as u32;
            let count = indices.len() as u32;
            grid_nodes.extend_from_slice(&indices);
            grid_cells.insert(key, (offset, count));
        }

        info!(
            "NeutronGraph ready in {}ms: {} nodes, {} grid cells ({:.0} LY cells). \
             Neighbours generated on-demand within the per-request boosted range (no edge cap).",
            t_start.elapsed().as_millis(),
            n,
            grid_cells.len(),
            cell_size
        );

        Ok(Self {
            x,
            y,
            z,
            id64,
            id_map,
            grid_cell_size: cell_size,
            grid_cells,
            grid_nodes,
            vk_nodes,
        })
    }

    pub fn len(&self) -> usize {
        self.id64.len()
    }

    /// On-demand neighbour generation: pushes the index of every neutron
    /// strictly within `range` of (cx,cy,cz) into `out`. The caller clears and
    /// reuses `out` across calls, so the A*/greedy hot loops allocate nothing
    /// and there is NO cap — the search sees the true reachability graph.
    fn collect_neighbors(&self, cx: f64, cy: f64, cz: f64, range: f64, out: &mut Vec<u32>) {
        let cs = self.grid_cell_size as f64;
        let bx = (cx / cs).floor() as i32;
        let by = (cy / cs).floor() as i32;
        let bz = (cz / cs).floor() as i32;
        let rsq = (range * range) as f32;
        let search_cells = (range / cs).ceil() as i32;

        for dx in -search_cells..=search_cells {
            for dy in -search_cells..=search_cells {
                for dz in -search_cells..=search_cells {
                    if let Some(&(off, cnt)) = self.grid_cells.get(&(bx + dx, by + dy, bz + dz)) {
                        let slice = &self.grid_nodes[off as usize..(off + cnt) as usize];
                        for &ni in slice {
                            let ddx = self.x[ni as usize] as f64 - cx;
                            let ddy = self.y[ni as usize] as f64 - cy;
                            let ddz = self.z[ni as usize] as f64 - cz;
                            let d2 = (ddx * ddx + ddy * ddy + ddz * ddz) as f32;
                            if d2 <= rsq && d2 > 0.0 {
                                out.push(ni);
                            }
                        }
                    }
                }
            }
        }
    }

    /// O(1) nearest neutron to a coordinate using the internal grid. Retained as
    /// public API for other handlers.
    pub fn nearest(&self, tx: f64, ty: f64, tz: f64, max_range: f64) -> Option<i64> {
        let cs = self.grid_cell_size as f64;
        let bx = (tx / cs).floor() as i32;
        let by = (ty / cs).floor() as i32;
        let bz = (tz / cs).floor() as i32;
        let mut best: Option<(u32, f64)> = None;
        let max_sq = max_range * max_range;
        let search_cells = (max_range / cs).ceil() as i32;

        for dx in -search_cells..=search_cells {
            for dy in -search_cells..=search_cells {
                for dz in -search_cells..=search_cells {
                    if let Some(&(off, cnt)) = self.grid_cells.get(&(bx + dx, by + dy, bz + dz)) {
                        let slice = &self.grid_nodes[off as usize..(off + cnt) as usize];
                        for &ni in slice {
                            let nx = self.x[ni as usize] as f64;
                            let ny = self.y[ni as usize] as f64;
                            let nz = self.z[ni as usize] as f64;
                            let d2 = (nx - tx).powi(2) + (ny - ty).powi(2) + (nz - tz).powi(2);
                            if d2 <= max_sq && d2 > 0.0 && best.map_or(true, |(_, bd)| d2 < bd) {
                                best = Some((ni, d2));
                            }
                        }
                    }
                }
            }
        }
        best.map(|(ni, _)| self.id64[ni as usize])
    }

    /// Allocating variant of collect_neighbors. Retained as public API for other
    /// handlers; the router itself uses collect_neighbors.
    pub fn neighbors_within(&self, cx: f64, cy: f64, cz: f64, range: f64) -> Vec<u32> {
        let mut out = Vec::new();
        self.collect_neighbors(cx, cy, cz, range, &mut out);
        out
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// A* heap node
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Eq, PartialEq)]
struct ANode {
    f_bits: u64,
    idx: u32,
    g: u32,
}

impl Ord for ANode {
    fn cmp(&self, o: &Self) -> Ordering {
        o.f_bits.cmp(&self.f_bits) // reversed => BinaryHeap pops smallest f
    }
}
impl PartialOrd for ANode {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

const SENTINEL: u32 = u32::MAX;

#[inline(always)]
fn dist_sq(x1: f64, y1: f64, z1: f64, x2: f64, y2: f64, z2: f64) -> f64 {
    (x2 - x1).powi(2) + (y2 - y1).powi(2) + (z2 - z1).powi(2)
}

/// Admissible min-hop heuristic: a lower bound on the number of remaining
/// boosted jumps from neutron `idx` to the destination.
#[inline]
fn h_to_dst(graph: &NeutronGraph, idx: u32, dx: f64, dy: f64, dz: f64, boosted_range: f64) -> u32 {
    let nx = graph.x[idx as usize] as f64;
    let ny = graph.y[idx as usize] as f64;
    let nz = graph.z[idx as usize] as f64;
    (dist_sq(nx, ny, nz, dx, dy, dz).sqrt() / boosted_range).ceil() as u32
}

/// Unified coordinate fetcher (checks resident graph first, then local normal stars)
fn get_node(
    graph: &NeutronGraph,
    normal_nodes: &HashMap<i64, (f64, f64, f64)>,
    id: i64,
) -> (f64, f64, f64, bool) {
    if let Some(&g_idx) = graph.id_map.get(&id) {
        (
            graph.x[g_idx as usize] as f64,
            graph.y[g_idx as usize] as f64,
            graph.z[g_idx as usize] as f64,
            true,
        )
    } else if let Some(&pos) = normal_nodes.get(&id) {
        (pos.0, pos.1, pos.2, false)
    } else {
        (0.0, 0.0, 0.0, false)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Main router
// ═══════════════════════════════════════════════════════════════════════════

async fn do_neutron_route(
    state: Arc<AppState>,
    params: NeutronRouteQuery,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let _permit = state.astar_semaphore.acquire().await.map_err(|_| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "Server overloaded — A* queue full".into(),
        )
    })?;

    let pool = state.db_pool.clone();
    let graph = {
        let guard = state.neutron_graph.read().await;
        match guard.as_ref() {
            Some(g) => g.clone(),
            None => {
                return Err((
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Resident neutron graph is still building at startup — try again shortly.".into(),
                ));
            }
        }
    };

    let result = tokio::task::spawn_blocking(move || -> Result<serde_json::Value, String> {
        let conn = pool.get().map_err(|e| e.to_string())?;
        let t_start = Instant::now();

        // ── Resolve source / destination ─────────────────────────────────────
        let get_sys = |input: &str| -> Result<(i64, String, f64, f64, f64), String> {
            if let Ok(id) = input.parse::<i64>() {
                conn.query_row(
                    "SELECT s.id64, s.name, i.minX, i.minY, i.minZ \
                     FROM systems s JOIN systems_index i ON s.id64=i.id \
                     WHERE s.id64=? LIMIT 1",
                    rusqlite::params![id],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
                )
                .map_err(|_| format!("System ID '{}' not found", input))
            } else {
                conn.query_row(
                    "SELECT s.id64, s.name, i.minX, i.minY, i.minZ \
                     FROM systems s JOIN systems_index i ON s.id64=i.id \
                     WHERE s.name=? COLLATE NOCASE LIMIT 1",
                    rusqlite::params![input],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
                )
                .map_err(|_| format!("System '{}' not found", input))
            }
        };

        let (src_id, _src_name, x1, y1, z1) = get_sys(&params.source)?;
        let (dst_id, _dst_name, x2, y2, z2) = get_sys(&params.destination)?;

        let total_distance = dist_sq(x1, y1, z1, x2, y2, z2).sqrt();
        let multiplier = if params.supercharge_type.to_lowercase() == "caspian" {
            6.0
        } else {
            4.0
        };
        let base_range = params.range;
        let boosted_range = base_range * multiplier;

        info!(
            "Neutron route: {} -> {}, {:.0} LY, base {:.2} LY, boosted {:.1} LY",
            params.source, params.destination, total_distance, base_range, boosted_range
        );

        // Normal (non-neutron) systems are stored request-locally; neutrons live
        // in the resident graph.
        let mut normal_nodes: HashMap<i64, (f64, f64, f64)> = HashMap::new();
        let src_is_neutron = graph.id_map.contains_key(&src_id);
        let dst_is_neutron = graph.id_map.contains_key(&dst_id);
        if !src_is_neutron {
            normal_nodes.insert(src_id, (x1, y1, z1));
        }
        if !dst_is_neutron {
            normal_nodes.insert(dst_id, (x2, y2, z2));
        }

        // ── Determine the route ──────────────────────────────────────────────
        let src_range = if src_is_neutron { boosted_range } else { base_range };

        let final_path: Vec<i64>;
        let mut is_optimal = false;

        if src_id == dst_id {
            final_path = vec![src_id];
        } else if dist_sq(x1, y1, z1, x2, y2, z2) <= src_range * src_range {
            final_path = vec![src_id, dst_id];
        } else {
            let mut normal_stmt = conn
                .prepare(
                    "SELECT i.id, i.minX, i.minY, i.minZ \
                     FROM systems_index i \
                     WHERE i.minX BETWEEN ? AND ? \
                       AND i.minY BETWEEN ? AND ? \
                       AND i.minZ BETWEEN ? AND ?",
                )
                .map_err(|e| e.to_string())?;

            // ── Phase 1: Greedy (upper bound + fallback) ─────────────────────
            let greedy_path = greedy_route(
                &graph,
                &mut normal_stmt,
                &mut normal_nodes,
                src_id,
                dst_id,
                x1, y1, z1,
                x2, y2, z2,
                base_range,
                boosted_range,
            );
            let greedy_jumps = greedy_path.as_ref().map(|p| (p.len() - 1) as u32);
            let mu_init = greedy_jumps.unwrap_or(u32::MAX);

            info!(
                "Greedy: {} ({}ms)",
                greedy_jumps.map_or("FAILED".to_string(), |j| format!("{} jumps", j)),
                t_start.elapsed().as_millis()
            );

            // ── Phase 2: A* with wide-radius entry seeding ───────────────────
            let use_astar =
                params.engine.as_deref().unwrap_or("astar").to_lowercase() != "greedy";

            let astar_result = if use_astar {
                astar_min_hop(
                    &graph,
                    src_id, dst_id,
                    x1, y1, z1,
                    x2, y2, z2,
                    base_range,
                    boosted_range,
                    mu_init,
                    &t_start,
                )
            } else {
                None
            };

            // ── Phase 3: If A* returned a neutron chain, build the full path ─
            //
            // astar_min_hop returns the pure neutron chain (entry neutron through
            // exit neutron). We need to splice in the actual bridges:
            //   source -> [entry bridge] -> [neutron chain] -> [exit] -> dest
            //
            // The A* used estimated entry costs (ceil(dist/base_range)) which may
            // undercount. The bridge walk computes the real hop count. If the
            // assembled path is still shorter than greedy, we use it.
            let assembled_astar: Option<Vec<i64>> = astar_result.and_then(|neutron_chain| {
                if neutron_chain.is_empty() {
                    return None;
                }

                let entry_id = neutron_chain[0];
                let exit_id = *neutron_chain.last().unwrap();

                // Entry bridge: source -> first neutron in chain.
                let entry_bridge: Vec<i64> = if src_id == entry_id {
                    // Source IS the entry neutron (src is neutron on-chain).
                    vec![src_id]
                } else if src_is_neutron {
                    // Source is a neutron but not the entry. One boosted hop.
                    vec![src_id]
                } else {
                    // Source is a normal star. Walk through normal stars to the
                    // entry neutron. This is the critical path that the old code
                    // couldn't compute because it only seeded from base_range.
                    let (ex, ey, ez, _) = get_node(&graph, &normal_nodes, entry_id);
                    match bridge_to_neutron(
                        &graph,
                        &mut normal_stmt,
                        &mut normal_nodes,
                        src_id, x1, y1, z1,
                        entry_id, ex, ey, ez,
                        base_range,
                        boosted_range,
                    ) {
                        Some(bridge) => bridge,
                        None => {
                            // Can't bridge to this neutron. Fall back.
                            return None;
                        }
                    }
                };

                // Assemble: bridge + chain + exit.
                let bridge_hops = entry_bridge.len().saturating_sub(1);
                let chain_hops = neutron_chain.len().saturating_sub(1);
                let mut full_path = entry_bridge;

                // Append neutron chain (skip the entry neutron if it's already
                // the last element of the bridge, avoiding a duplicate).
                let chain_start = if full_path.last().copied() == Some(entry_id) {
                    1
                } else {
                    0
                };
                full_path.extend_from_slice(&neutron_chain[chain_start..]);

                // Append destination if it's not already the exit neutron.
                if exit_id != dst_id {
                    full_path.push(dst_id);
                }

                let assembled_jumps = (full_path.len() - 1) as u32;
                info!(
                    "A* assembled: {} jumps (bridge={}, chain={}) in {}ms",
                    assembled_jumps, bridge_hops, chain_hops,
                    t_start.elapsed().as_millis()
                );

                // Only use if it actually beats greedy.
                if greedy_jumps.map_or(true, |gj| assembled_jumps < gj) {
                    Some(full_path)
                } else if greedy_jumps.map_or(false, |gj| assembled_jumps == gj) {
                    // Same length but A* chain is all-neutron (better for the player).
                    Some(full_path)
                } else {
                    None
                }
            });

            match (assembled_astar, greedy_path) {
                (Some(a), Some(gp)) => {
                    if a.len() <= gp.len() {
                        is_optimal = a.len() < gp.len();
                        final_path = a;
                    } else {
                        final_path = gp;
                    }
                }
                (Some(a), None) => {
                    is_optimal = true;
                    final_path = a;
                }
                (None, Some(gp)) => {
                    final_path = gp;
                }
                (None, None) => {
                    return Err(
                        "No route found: destination unreachable at this jump range. \
                         Try a ship with longer range."
                            .into(),
                    );
                }
            }
        }

        info!(
            "Final route: {} jumps in {}ms (improved on greedy: {})",
            final_path.len().saturating_sub(1),
            t_start.elapsed().as_millis(),
            is_optimal
        );

        // ── JSON generation ──────────────────────────────────────────────────
        let path_names = batch_resolve_names(&conn, &final_path)?;
        let mut route_json = Vec::with_capacity(final_path.len());
        let mut dist_from_start = 0.0f64;

        for step in 0..final_path.len() {
            let nid = final_path[step];
            let (nx, ny, nz, is_n) = get_node(&graph, &normal_nodes, nid);
            let nname = path_names.get(&nid).cloned().unwrap_or_default();
            let d_dest = dist_sq(nx, ny, nz, x2, y2, z2).sqrt();

            let (jdist, used_boost) = if step + 1 < final_path.len() {
                let next_id = final_path[step + 1];
                let (nnx, nny, nnz, _) = get_node(&graph, &normal_nodes, next_id);
                (dist_sq(nx, ny, nz, nnx, nny, nnz).sqrt(), is_n)
            } else {
                (0.0, false)
            };

            route_json.push(serde_json::json!({
                "system": nname,
                "id64": nid.to_string(),
                "distance_from_start": (dist_from_start * 100.0).round() / 100.0,
                "distance_to_destination": (d_dest * 100.0).round() / 100.0,
                "jump_distance": (jdist * 100.0).round() / 100.0,
                "used_neutron_boost": used_boost,
                "is_neutron": is_n,
            }));
            dist_from_start += jdist;
        }

        Ok(serde_json::json!({
            "source": params.source,
            "destination": params.destination,
            "total_distance_ly": (total_distance * 100.0).round() / 100.0,
            "totalJumps": route_json.len().saturating_sub(1),
            "optimised": is_optimal,
            "route": route_json,
        }))
    })
    .await
    .unwrap()
    .map_err(|e| (StatusCode::BAD_REQUEST, e))?;

    Ok(Json(result))
}

// ═══════════════════════════════════════════════════════════════════════════
// A* with wide-radius entry seeding.
//
// All neutron-to-neutron edges have unit cost and exist iff the pair is within
// boosted_range, so min-jump == unit-cost shortest path. The heuristic h_to_dst
// is admissible, making the neutron-chain portion of the search optimal.
//
// When the source is NOT a neutron, the old code seeded from neutrons within
// one base_range hop. That misses every neutron further than 75 LY, which is
// where the good highway entry points are. The new seeding searches a wide
// radius (up to boosted_range or 15x base_range) and estimates entry cost as
// ceil(euclidean_dist / base_range). This is a lower bound on actual base hops,
// so the chain the A* picks is the best *given optimistic entry estimates*.
//
// The caller (do_neutron_route) then computes the ACTUAL entry bridge through
// normal stars and verifies the assembled total against the greedy fallback.
//
// Returns: the pure neutron chain (entry neutron through exit neutron), or None.
//          Does NOT include src_id or dst_id; the caller handles bridging.
// ═══════════════════════════════════════════════════════════════════════════

#[allow(clippy::too_many_arguments)]
fn astar_min_hop(
    graph: &NeutronGraph,
    src_id: i64,
    dst_id: i64,
    sx: f64,
    sy: f64,
    sz: f64,
    dx: f64,
    dy: f64,
    dz: f64,
    base_range: f64,
    boosted_range: f64,
    mu_init: u32,
    t_start: &Instant,
) -> Option<Vec<i64>> {
    let n = graph.len();
    let goal = n as u32;
    let src_idx_opt = graph.id_map.get(&src_id).copied();
    let dst_idx_opt = graph.id_map.get(&dst_id).copied();
    let dst_is_neutron = dst_idx_opt.is_some();
    let boosted_sq = boosted_range * boosted_range;

    let mut g: Vec<u32> = vec![SENTINEL; n + 1];
    let mut came: Vec<u32> = vec![SENTINEL; n + 1];
    let mut closed: Vec<bool> = vec![false; n + 1];

    let mut open: BinaryHeap<ANode> = BinaryHeap::with_capacity(8192);
    let mut mu = mu_init;

    let mut nbr: Vec<u32> = Vec::with_capacity(256);

    // ── Seed ─────────────────────────────────────────────────────────────────
    if let Some(si) = src_idx_opt {
        // Source IS a neutron: start at g=0.
        g[si as usize] = 0;
        came[si as usize] = SENTINEL;
        open.push(ANode {
            f_bits: (h_to_dst(graph, si, dx, dy, dz, boosted_range) as f64).to_bits(),
            idx: si,
            g: 0,
        });
    } else {
        // Source is a normal star. Seed from ALL neutrons within a wide radius,
        // estimating entry cost as ceil(dist / base_range). This lets the A*
        // evaluate entry points the old single-hop seeding couldn't see.
        //
        // ceil(dist/base) is a LOWER BOUND on actual base-range hops (stars
        // aren't on a perfect line). The caller verifies the real cost after
        // the A* picks the best chain. The mu pruning ensures we only push
        // seeds that can plausibly beat the greedy upper bound.
        let entry_radius = boosted_range.max(base_range * 15.0);
        nbr.clear();
        graph.collect_neighbors(sx, sy, sz, entry_radius, &mut nbr);

        let mut seed_count = 0u32;
        for &ni in &nbr {
            let nx = graph.x[ni as usize] as f64;
            let ny = graph.y[ni as usize] as f64;
            let nz = graph.z[ni as usize] as f64;
            let dist = dist_sq(sx, sy, sz, nx, ny, nz).sqrt();
            let entry_hops = (dist / base_range).ceil() as u32;

            let hh = h_to_dst(graph, ni, dx, dy, dz, boosted_range);
            if entry_hops + hh >= mu {
                continue; // pruned: can't beat greedy even with optimistic entry
            }

            if entry_hops < g[ni as usize] {
                g[ni as usize] = entry_hops;
                came[ni as usize] = SENTINEL;
                open.push(ANode {
                    f_bits: ((entry_hops + hh) as f64).to_bits(),
                    idx: ni,
                    g: entry_hops,
                });
                seed_count += 1;
            }
        }
        info!("A* seeded {} entry neutrons within {:.0} LY", seed_count, entry_radius);
    }

    // ── Main loop ────────────────────────────────────────────────────────────
    while let Some(node) = open.pop() {
        if t_start.elapsed().as_millis() > REFINE_BUDGET_MS {
            break;
        }
        let idx = node.idx;
        let gg = node.g;

        if idx == goal {
            break;
        }
        if closed[idx as usize] {
            continue;
        }
        if gg > g[idx as usize] {
            continue;
        }
        if gg + h_to_dst(graph, idx, dx, dy, dz, boosted_range) >= mu {
            break;
        }
        closed[idx as usize] = true;

        let cx = graph.x[idx as usize] as f64;
        let cy = graph.y[idx as usize] as f64;
        let cz = graph.z[idx as usize] as f64;

        // Edge to the virtual goal (the final hop to the destination).
        let goal_cost: Option<u32> = if dst_is_neutron {
            if idx == dst_idx_opt.unwrap() {
                Some(0)
            } else {
                None
            }
        } else if dist_sq(cx, cy, cz, dx, dy, dz) <= boosted_sq {
            Some(1)
        } else {
            None
        };
        if let Some(c) = goal_cost {
            let tg = gg + c;
            if tg < g[goal as usize] {
                g[goal as usize] = tg;
                came[goal as usize] = idx;
                if tg < mu {
                    mu = tg;
                }
                open.push(ANode {
                    f_bits: (tg as f64).to_bits(),
                    idx: goal,
                    g: tg,
                });
            }
        }

        // Neutron neighbours within the true boosted range (no cap).
        nbr.clear();
        graph.collect_neighbors(cx, cy, cz, boosted_range, &mut nbr);
        for &ni in &nbr {
            if closed[ni as usize] {
                continue;
            }
            let tg = gg + 1;
            if tg < g[ni as usize] {
                g[ni as usize] = tg;
                came[ni as usize] = idx;
                let hh = h_to_dst(graph, ni, dx, dy, dz, boosted_range);
                open.push(ANode {
                    f_bits: ((tg + hh) as f64).to_bits(),
                    idx: ni,
                    g: tg,
                });
            }
        }
    }

    if g[goal as usize] == SENTINEL {
        return None;
    }

    // ── Reconstruct the pure neutron chain ────────────────────────────────────
    // Walk came_from from the goal's predecessor back to a seed. The result is
    // the neutron-only chain; the caller handles source/destination bridging.
    let mut chain: Vec<u32> = Vec::new();
    let mut cur = came[goal as usize];
    let mut steps = 0usize;
    while cur != SENTINEL {
        chain.push(cur);
        cur = came[cur as usize];
        steps += 1;
        if steps > n + 2 {
            return None;
        }
    }
    chain.reverse();

    let path: Vec<i64> = chain.iter().map(|&i| graph.id64[i as usize]).collect();
    if path.is_empty() {
        return None;
    }

    info!(
        "A* chain: {} neutron hops, estimated total {} jumps",
        path.len().saturating_sub(1),
        g[goal as usize]
    );
    Some(path)
}

// ═══════════════════════════════════════════════════════════════════════════
// Bridge from a normal star to a specific neutron through normal stars.
//
// Greedy walk from (sx,sy,sz) toward the target neutron (tx,ty,tz), using
// base_range hops through the systems_index table. At each step, also checks
// if any in-range neutron is closer to the target (to hop onto the neutron
// grid early if possible). Terminates when the target is within jump range.
//
// Returns the bridge path INCLUDING the source but EXCLUDING the target
// (the caller appends the target as part of the neutron chain).
// ═══════════════════════════════════════════════════════════════════════════

#[allow(clippy::too_many_arguments)]
fn bridge_to_neutron(
    graph: &NeutronGraph,
    normal_stmt: &mut rusqlite::Statement,
    normal_nodes: &mut HashMap<i64, (f64, f64, f64)>,
    src_id: i64,
    sx: f64,
    sy: f64,
    sz: f64,
    target_id: i64,
    tx: f64,
    ty: f64,
    tz: f64,
    base_range: f64,
    boosted_range: f64,
) -> Option<Vec<i64>> {
    let mut path = vec![src_id];
    let mut visited: HashSet<i64> = HashSet::new();
    visited.insert(src_id);
    let (mut cx, mut cy, mut cz) = (sx, sy, sz);
    let mut cur_id = src_id;
    let mut cur_is_neutron = graph.id_map.contains_key(&src_id);
    let mut nbr: Vec<u32> = Vec::with_capacity(256);

    for _ in 0..500usize {
        let cur_range = if cur_is_neutron { boosted_range } else { base_range };
        let d_target = dist_sq(cx, cy, cz, tx, ty, tz).sqrt();

        // Can we reach the target directly?
        if d_target <= cur_range {
            return Some(path);
        }

        // Look for neutrons within range that are closer to the target.
        // Prioritise the target itself, then any neutron making progress.
        nbr.clear();
        graph.collect_neighbors(cx, cy, cz, cur_range, &mut nbr);

        let mut best_neutron_id: Option<i64> = None;
        let mut best_neutron_pos = (0.0f64, 0.0f64, 0.0f64);
        let mut best_neutron_d = d_target;

        for &ni in &nbr {
            let nid = graph.id64[ni as usize];
            if visited.contains(&nid) {
                continue;
            }
            let nx = graph.x[ni as usize] as f64;
            let ny = graph.y[ni as usize] as f64;
            let nz = graph.z[ni as usize] as f64;
            let nd = dist_sq(nx, ny, nz, tx, ty, tz).sqrt();
            if nd < best_neutron_d {
                best_neutron_d = nd;
                best_neutron_id = Some(nid);
                best_neutron_pos = (nx, ny, nz);
            }
        }

        if let Some(nid) = best_neutron_id {
            visited.insert(nid);
            path.push(nid);
            cur_id = nid;
            cx = best_neutron_pos.0;
            cy = best_neutron_pos.1;
            cz = best_neutron_pos.2;
            cur_is_neutron = true;
            continue;
        }

        // No useful neutron. Step through a normal star toward the target.
        let moved = attempt_normal_move_targeted(
            normal_stmt,
            normal_nodes,
            &mut visited,
            &mut path,
            &mut cur_id,
            &mut cx,
            &mut cy,
            &mut cz,
            tx, ty, tz,
            cur_range,
        )
        .ok()?;

        if !moved {
            return None;
        }
        cur_is_neutron = graph.id_map.contains_key(&cur_id);
    }
    None
}

// ═══════════════════════════════════════════════════════════════════════════
// Normal-star step toward a specific target. Shared by bridge_to_neutron and
// the greedy fallback. Picks the in-range normal star closest to (tx,ty,tz).
// ═══════════════════════════════════════════════════════════════════════════

#[allow(clippy::too_many_arguments)]
fn attempt_normal_move_targeted(
    stmt: &mut rusqlite::Statement,
    normal_nodes: &mut HashMap<i64, (f64, f64, f64)>,
    visited: &mut HashSet<i64>,
    path: &mut Vec<i64>,
    cur_id: &mut i64,
    cx: &mut f64,
    cy: &mut f64,
    cz: &mut f64,
    tx: f64,
    ty: f64,
    tz: f64,
    range: f64,
) -> Result<bool, String> {
    let d_target = dist_sq(*cx, *cy, *cz, tx, ty, tz);

    for &r in &[range, range * 1.05] {
        let rows: Vec<(i64, f64, f64, f64)> = stmt
            .query_map(
                rusqlite::params![*cx - r, *cx + r, *cy - r, *cy + r, *cz - r, *cz + r],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, f64>(1)?,
                        row.get::<_, f64>(2)?,
                        row.get::<_, f64>(3)?,
                    ))
                },
            )
            .map_err(|e| e.to_string())?
            .filter_map(|r| r.ok())
            .filter(|&(nid, nx, ny, nz)| {
                if nid == *cur_id || visited.contains(&nid) {
                    return false;
                }
                let d2 = dist_sq(nx, ny, nz, *cx, *cy, *cz);
                d2 <= r * r && d2 > 0.0
            })
            .collect();

        let best = rows
            .iter()
            .filter(|&&(_, nx, ny, nz)| dist_sq(nx, ny, nz, tx, ty, tz) < d_target)
            .min_by(|a, b| {
                let da = dist_sq(a.1, a.2, a.3, tx, ty, tz);
                let db = dist_sq(b.1, b.2, b.3, tx, ty, tz);
                da.partial_cmp(&db).unwrap_or(Ordering::Equal)
            });

        if let Some(&(nid, nx, ny, nz)) = best {
            normal_nodes.insert(nid, (nx, ny, nz));
            visited.insert(nid);
            path.push(nid);
            *cur_id = nid;
            *cx = nx;
            *cy = ny;
            *cz = nz;
            return Ok(true);
        }
    }
    Ok(false)
}

// ═══════════════════════════════════════════════════════════════════════════
// Greedy best-first route toward the destination.
//
// Always jumps to the in-range neutron closest to the destination. Doubles as
// the A* upper bound and as the normal-star-capable fallback for neutron-sparse
// regions. On-demand neighbours, no edge cap.
// ═══════════════════════════════════════════════════════════════════════════

#[allow(clippy::too_many_arguments)]
fn greedy_route(
    graph: &NeutronGraph,
    normal_stmt: &mut rusqlite::Statement,
    normal_nodes: &mut HashMap<i64, (f64, f64, f64)>,
    src_id: i64,
    dst_id: i64,
    sx: f64,
    sy: f64,
    sz: f64,
    dx: f64,
    dy: f64,
    dz: f64,
    base_range: f64,
    boosted_range: f64,
) -> Option<Vec<i64>> {
    let mut path = vec![src_id];
    let mut visited: HashSet<i64> = HashSet::new();
    visited.insert(src_id);
    let mut cur_id = src_id;
    let (mut cx, mut cy, mut cz) = (sx, sy, sz);
    let mut cur_is_neutron = graph.id_map.contains_key(&src_id);
    let mut nbr: Vec<u32> = Vec::with_capacity(256);

    for _ in 0..200_000usize {
        let cur_range = if cur_is_neutron { boosted_range } else { base_range };
        let d_dst = dist_sq(cx, cy, cz, dx, dy, dz).sqrt();
        if d_dst <= cur_range {
            path.push(dst_id);
            return Some(path);
        }

        // Best in-range neutron, closest to the destination, strictly progressing.
        let mut best_id: Option<i64> = None;
        let mut best_pos = (0.0f64, 0.0f64, 0.0f64);
        let mut best_d = d_dst;
        nbr.clear();
        graph.collect_neighbors(cx, cy, cz, cur_range, &mut nbr);
        for &ni in &nbr {
            let nid = graph.id64[ni as usize];
            if visited.contains(&nid) {
                continue;
            }
            let nx = graph.x[ni as usize] as f64;
            let ny = graph.y[ni as usize] as f64;
            let nz = graph.z[ni as usize] as f64;
            let nd = dist_sq(nx, ny, nz, dx, dy, dz).sqrt();
            if nd < best_d {
                best_d = nd;
                best_id = Some(nid);
                best_pos = (nx, ny, nz);
            }
        }

        if let Some(nid) = best_id {
            visited.insert(nid);
            path.push(nid);
            cur_id = nid;
            cx = best_pos.0;
            cy = best_pos.1;
            cz = best_pos.2;
            cur_is_neutron = true;
            continue;
        }

        // Sparse region: bridge across a normal star.
        let moved = attempt_normal_move_targeted(
            normal_stmt,
            normal_nodes,
            &mut visited,
            &mut path,
            &mut cur_id,
            &mut cx,
            &mut cy,
            &mut cz,
            dx,
            dy,
            dz,
            cur_range,
        )
        .ok()?;
        if !moved {
            return None;
        }
        cur_is_neutron = false;
    }
    None
}

// ═══════════════════════════════════════════════════════════════════════════
// Batch DB Resolution
// ═══════════════════════════════════════════════════════════════════════════

fn batch_resolve_names(
    conn: &rusqlite::Connection,
    path: &[i64],
) -> Result<HashMap<i64, String>, String> {
    let mut names: HashMap<i64, String> = HashMap::with_capacity(path.len());

    for chunk in path.chunks(500) {
        let placeholders: String = chunk.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!("SELECT id64, name FROM systems WHERE id64 IN ({})", placeholders);
        let mut stmt = conn.prepare(&sql).map_err(|e| e.to_string())?;

        let params: Vec<&dyn rusqlite::ToSql> = chunk
            .iter()
            .map(|id| id as &dyn rusqlite::ToSql)
            .collect();

        let rows = stmt
            .query_map(params.as_slice(), |r| {
                Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
            })
            .map_err(|e| e.to_string())?;

        for row in rows {
            if let Ok((id, name)) = row {
                names.insert(id, name);
            }
        }
    }
    Ok(names)
}

// ═══════════════════════════════════════════════════════════════════════════
// Handler
// ═══════════════════════════════════════════════════════════════════════════

pub async fn neutron_route_post(
    State(state): State<Arc<AppState>>,
    Json(params): Json<NeutronRouteQuery>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    do_neutron_route(state, params).await
}

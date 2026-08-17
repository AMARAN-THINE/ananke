use axum::{
    extract::{Query, State},
    http::StatusCode,
    Json,
};
use serde::Deserialize;
use std::sync::Arc;

use crate::db::current_time_secs;
use crate::state::AppState;

/// Cached bubble progress result.
pub struct BubbleCacheEntry {
    pub data: serde_json::Value,
    pub expires_at: u64,
}

#[derive(Deserialize)]
pub struct BubbleQuery {
    /// Capital system name (case-insensitive).
    pub capital: String,
    /// Search radius in LY. Default 250, max 500.
    pub radius: Option<f64>,
    /// Ring width in LY. Default 50, min 10.
    pub ring_size: Option<f64>,
}

/// GET /api/bubble-progress?capital=SomeSystem&radius=250&ring_size=50
///
/// Returns colonisation progress around a capital system. The target set
/// is every system within `radius` LY that has at least one colonisation-
/// worthy body (Earth-like, Water world, Ammonia world, or any body with
/// terraformingState = Terraformable/Terraforming completed). Systems are
/// tier-weighted so colonising high-value targets moves the bar more than
/// settling a random terraformable HMC.
///
/// Cached for 1 hour per unique (capital, radius, ring_size) triple.
pub async fn get_bubble_progress(
    State(state): State<Arc<AppState>>,
    Query(params): Query<BubbleQuery>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let radius = params.radius.unwrap_or(250.0).min(500.0);
    let ring_size = params.ring_size.unwrap_or(50.0).max(10.0);

    let cache_key = format!(
        "{}|{}|{}",
        params.capital.to_lowercase(),
        radius as i64,
        ring_size as i64,
    );

    // Return cached result if fresh
    let now = current_time_secs();
    {
        let cache = state.bubble_cache.lock().await;
        if let Some(entry) = cache.get(&cache_key) {
            if now < entry.expires_at {
                return Ok(Json(entry.data.clone()));
            }
        }
    }

    let _permit = state
        .query_semaphore
        .acquire()
        .await
        .map_err(|_| (StatusCode::SERVICE_UNAVAILABLE, "Server overloaded".into()))?;

    let pool = state.db_pool.clone();
    let capital = params.capital.clone();

    let result = tokio::task::spawn_blocking(move || -> Result<serde_json::Value, String> {
        let conn = pool.get().map_err(|e| e.to_string())?;

        // ── 1. Resolve capital coords ──────────────────────────────────
        let (cx, cy, cz, cap_id64, cap_name): (f64, f64, f64, i64, String) = conn
            .query_row(
                "SELECT i.minX, i.minY, i.minZ, s.id64, s.name \
                 FROM systems s JOIN systems_index i ON s.id64 = i.id \
                 WHERE s.name = ?1 COLLATE NOCASE LIMIT 1",
                rusqlite::params![capital],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
            )
            .map_err(|_| format!("Capital system '{}' not found", capital))?;

        // Bounding cube for the R-tree pre-filter
        let min_x = cx - radius;
        let max_x = cx + radius;
        let min_y = cy - radius;
        let max_y = cy + radius;
        let min_z = cz - radius;
        let max_z = cz + radius;

        // ── 2. Pull every colonisation-worthy system in one shot ───────
        //
        // The JOIN + GROUP BY collapses per-body rows into one row per
        // system with tier indicator flags. The R-tree narrows the scan
        // to a cube; we sphere-filter in Rust afterwards.
        let mut stmt = conn
            .prepare(
                "SELECT s.id64, s.name, s.population,
                        i.minX, i.minY, i.minZ,
                        GROUP_CONCAT(DISTINCT b.subType) AS body_types,
                        MAX(CASE WHEN b.subType IN ('Earth-like world','Earthlike body')
                                 THEN 1 ELSE 0 END)                          AS has_elw,
                        MAX(CASE WHEN b.subType = 'Water world'
                                 THEN 1 ELSE 0 END)                          AS has_ww,
                        MAX(CASE WHEN b.subType = 'Ammonia world'
                                 THEN 1 ELSE 0 END)                          AS has_aw
                 FROM systems_index i
                 JOIN systems s  ON i.id = s.id64
                 JOIN bodies  b  ON s.id64 = b.systemId64
                 WHERE i.minX >= ?1 AND i.maxX <= ?2
                   AND i.minY >= ?3 AND i.maxY <= ?4
                   AND i.minZ >= ?5 AND i.maxZ <= ?6
                   AND (
                       b.subType IN ('Earth-like world','Earthlike body',
                                     'Water world','Ammonia world')
                       OR b.terraformingState IN ('Terraformable',
                                                  'Terraforming completed')
                   )
                 GROUP BY s.id64",
            )
            .map_err(|e| e.to_string())?;

        struct SysEntry {
            id64: i64,
            name: String,
            population: i64,
            distance_ly: f64,
            tier: u8,
            tier_label: &'static str,
            weight: u32,
            body_types: String,
        }

        let radius_sq = radius * radius;
        let mut systems: Vec<SysEntry> = Vec::new();

        let rows = stmt
            .query_map(
                rusqlite::params![min_x, max_x, min_y, max_y, min_z, max_z],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,            // id64
                        row.get::<_, String>(1)?,         // name
                        row.get::<_, i64>(2)?,            // population
                        row.get::<_, f64>(3)?,            // x
                        row.get::<_, f64>(4)?,            // y
                        row.get::<_, f64>(5)?,            // z
                        row.get::<_, Option<String>>(6)?, // body_types
                        row.get::<_, i32>(7)?,            // has_elw
                        row.get::<_, i32>(8)?,            // has_ww
                        row.get::<_, i32>(9)?,            // has_aw
                    ))
                },
            )
            .map_err(|e| e.to_string())?;

        for row in rows {
            let (id64, name, pop, x, y, z, body_types, has_elw, has_ww, has_aw) =
                row.map_err(|e| e.to_string())?;

            let dx = x - cx;
            let dy = y - cy;
            let dz = z - cz;
            let dist_sq = dx * dx + dy * dy + dz * dz;

            // Sphere clip
            if dist_sq > radius_sq {
                continue;
            }

            let distance_ly = dist_sq.sqrt();

            // Tier: best body type wins
            let (tier, tier_label, weight): (u8, &'static str, u32) = if has_elw > 0 {
                (3, "Earth-like", 5)
            } else if has_ww > 0 || has_aw > 0 {
                (2, "Water/Ammonia", 3)
            } else {
                (1, "Terraformable", 1)
            };

            systems.push(SysEntry {
                id64,
                name,
                population: pop,
                distance_ly: (distance_ly * 100.0).round() / 100.0,
                tier,
                tier_label,
                weight,
                body_types: body_types.unwrap_or_default(),
            });
        }

        systems.sort_by(|a, b| {
            a.distance_ly
                .partial_cmp(&b.distance_ly)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // ── 3. Bucket into concentric rings ────────────────────────────
        let ring_count = (radius / ring_size).ceil() as usize;
        let mut rings: Vec<serde_json::Value> = Vec::with_capacity(ring_count);

        for i in 0..ring_count {
            let inner = i as f64 * ring_size;
            let outer = ((i + 1) as f64 * ring_size).min(radius);

            let in_ring: Vec<&SysEntry> = systems
                .iter()
                .filter(|s| s.distance_ly >= inner && s.distance_ly < outer)
                .collect();

            let target = in_ring.len() as u32;
            let inhabited = in_ring.iter().filter(|s| s.population > 0).count() as u32;
            let w_max: u32 = in_ring.iter().map(|s| s.weight).sum();
            let w_score: u32 = in_ring
                .iter()
                .filter(|s| s.population > 0)
                .map(|s| s.weight)
                .sum();

            let system_list: Vec<serde_json::Value> = in_ring
                .iter()
                .map(|s| {
                    serde_json::json!({
                        "id64":       s.id64.to_string(),
                        "name":       s.name,
                        "distance_ly": s.distance_ly,
                        "population": s.population,
                        "inhabited":  s.population > 0,
                        "tier":       s.tier,
                        "tier_label": s.tier_label,
                        "weight":     s.weight,
                        "body_types": s.body_types,
                    })
                })
                .collect();

            rings.push(serde_json::json!({
                "inner_ly":          inner,
                "outer_ly":          outer,
                "target_systems":    target,
                "inhabited_systems": inhabited,
                "raw_progress":      ratio(inhabited, target),
                "weighted_score":    w_score,
                "weighted_max":      w_max,
                "weighted_progress": ratio(w_score, w_max),
                "systems":           system_list,
            }));
        }

        // ── 4. Aggregate totals ────────────────────────────────────────
        let total_target = systems.len() as u32;
        let total_inhabited = systems.iter().filter(|s| s.population > 0).count() as u32;
        let total_w_max: u32 = systems.iter().map(|s| s.weight).sum();
        let total_w_score: u32 = systems
            .iter()
            .filter(|s| s.population > 0)
            .map(|s| s.weight)
            .sum();

        let tier_stats = |t: u8| {
            let count = systems.iter().filter(|s| s.tier == t).count() as u32;
            let inh = systems
                .iter()
                .filter(|s| s.tier == t && s.population > 0)
                .count() as u32;
            serde_json::json!({ "target": count, "inhabited": inh })
        };

        Ok(serde_json::json!({
            "capital": {
                "id64":   cap_id64.to_string(),
                "name":   cap_name,
                "coords": { "x": cx, "y": cy, "z": cz },
            },
            "radius_ly":   radius,
            "ring_size_ly": ring_size,
            "overall": {
                "target_systems":    total_target,
                "inhabited_systems": total_inhabited,
                "raw_progress":      ratio(total_inhabited, total_target),
                "weighted_score":    total_w_score,
                "weighted_max":      total_w_max,
                "weighted_progress": ratio(total_w_score, total_w_max),
            },
            "tiers": {
                "t3_earthlike":       tier_stats(3),
                "t2_water_ammonia":   tier_stats(2),
                "t1_terraformable":   tier_stats(1),
            },
            "rings": rings,
        }))
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| {
        if e.contains("not found") {
            (StatusCode::NOT_FOUND, e)
        } else {
            (StatusCode::INTERNAL_SERVER_ERROR, e)
        }
    })?;

    // Cache for 1 hour
    {
        let mut cache = state.bubble_cache.lock().await;
        // Cap cache size to prevent unbounded growth
        if cache.len() >= 16 {
            // Evict the entry closest to expiry
            if let Some(oldest_key) = cache
                .iter()
                .min_by_key(|(_, v)| v.expires_at)
                .map(|(k, _)| k.clone())
            {
                cache.remove(&oldest_key);
            }
        }
        cache.insert(
            cache_key,
            BubbleCacheEntry {
                data: result.clone(),
                expires_at: now + 3600,
            },
        );
    }

    Ok(Json(result))
}

/// Safe ratio: returns 0.0 when denominator is 0, otherwise rounds to 4 dp.
fn ratio(num: u32, den: u32) -> f64 {
    if den == 0 {
        0.0
    } else {
        (num as f64 / den as f64 * 10000.0).round() / 10000.0
    }
}

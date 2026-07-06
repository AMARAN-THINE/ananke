use axum::{
    extract::{Query, State},
    http::StatusCode,
    Json,
};
use serde::Deserialize;
use std::sync::Arc;

use crate::procgen;
use crate::state::AppState;

#[derive(Deserialize)]
pub struct EstimateQuery {
    #[serde(rename = "systemName", alias = "name")]
    pub system_name: Option<String>,
}

pub async fn estimate_system(
    State(_state): State<Arc<AppState>>,
    Query(params): Query<EstimateQuery>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let name = params.system_name.ok_or_else(|| {
        (StatusCode::BAD_REQUEST, Json(serde_json::json!({
            "error": "Missing systemName parameter"
        })))
    })?;

    let pg = procgen::parse_procgen_name(&name).ok_or_else(|| {
        (StatusCode::BAD_REQUEST, Json(serde_json::json!({
            "error": "Not a procedurally generated system name",
            "name": name
        })))
    })?;

    let est = procgen::estimate_coords(&pg).ok_or_else(|| {
        (StatusCode::NOT_FOUND, Json(serde_json::json!({
            "error": "Could not decode sector name",
            "sector": pg.sector_name,
            "name": name
        })))
    })?;

    Ok(Json(serde_json::json!({
        "name": name,
        "estimatedCoords": {
            "x": (est.x * 100.0).round() / 100.0,
            "y": (est.y * 100.0).round() / 100.0,
            "z": (est.z * 100.0).round() / 100.0
        },
        "uncertaintyLy": est.uncertainty_ly,
        "massCode": pg.mass_char.to_string(),
        "sector": pg.sector_name,
        "source": "procgen_decode"
    })))
}

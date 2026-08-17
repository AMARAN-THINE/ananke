use crossbeam_channel::Sender;
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use std::collections::HashMap;
use std::sync::{atomic::AtomicU64, Arc};
use tokio::sync::{Mutex, RwLock, Semaphore};

use crate::handlers::bubble::BubbleCacheEntry;
use crate::handlers::neutron_route::NeutronGraph;
use crate::heatmap::Heatmap;
use crate::models::SpanshSystem;
use crate::vulkan_astar::VulkanAstar;

#[allow(dead_code)]
pub struct AppState {
    pub db_pool: Pool<SqliteConnectionManager>,
    pub query_semaphore: Arc<Semaphore>,
    pub astar_semaphore: Arc<Semaphore>,
    pub carrier_cache: Mutex<CarrierCache>,
    pub bubble_cache: Mutex<HashMap<String, BubbleCacheEntry>>,
    pub edmc_sender: Sender<Vec<SpanshSystem>>,
    pub edmc_api_key: Option<String>,
    pub edmc_stats: Arc<EdmcStats>,
    pub eddn_stats: Arc<EddnStats>,
    pub heatmap: Arc<Heatmap>,
    pub vulkan_astar: Option<Arc<VulkanAstar>>,
    /// Resident in-memory CSR neutron graph. None until the background
    /// build task finishes — handlers must check for None and return 503
    /// rather than block, so the rest of the API stays up during the build.
    pub neutron_graph: Arc<RwLock<Option<Arc<NeutronGraph>>>>,
    /// Primary-only neutron graph: only systems where the neutron star is
    /// the arrival star (< 100 Ls). Subset of the full graph — used when
    /// the player enables "primary stars only" to avoid 100k+ Ls detours.
    pub primary_neutron_graph: Arc<RwLock<Option<Arc<NeutronGraph>>>>,
}

#[allow(dead_code)]
pub struct CarrierCache {
    pub data: Option<serde_json::Value>,
    pub expires_at: u64,
}

/// Live ingest counters for EDMC
pub struct EdmcStats {
    pub systems_ingested: AtomicU64,
    pub bodies_ingested: AtomicU64,
    pub stations_ingested: AtomicU64,
    pub last_ingest_time: AtomicU64,
}

/// Live ingest counters for EDDN
pub struct EddnStats {
    pub messages_received: AtomicU64,
    pub messages_processed: AtomicU64,
    pub messages_dropped: AtomicU64,
    pub systems_emitted: AtomicU64,
    pub bodies_emitted: AtomicU64,
    pub stations_emitted: AtomicU64,
    pub last_message_time: AtomicU64,
    pub reconnects: AtomicU64,
    pub connected: AtomicU64,
}

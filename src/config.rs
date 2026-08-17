// --- CONFIGURATION ---
pub const DB_FILE: &str = "edsm_cube.db";
pub const PORT: u16 = 8000;
pub const URL_SYSTEMS_1DAY: &str = "https://downloads.spansh.co.uk/galaxy_1day.json.gz";
pub const FILE_SYSTEMS_1DAY: &str = "galaxy_1day.json.gz";
pub const FILE_SYSTEMS_DOWNLOADING: &str = "galaxy_1day.json.gz.downloading";
pub const SYNC_INTERVAL_SECONDS: u64 = 21600; // 6 hours
pub const MAX_CONCURRENT_QUERIES: usize = 6;
pub const MAX_CONCURRENT_ASTAR: usize = 2;
pub const SHIP_ROUTE_BUDGET_MS: u128 = 120_000; // 2 minutes
pub const EDMC_KEY_ENV: &str = "ANANKE_EDMC_KEY";

// --- EDDN ---
pub const EDDN_RELAY_URL: &str = "tcp://eddn.edcd.io:9500";
pub const EDDN_RELAY_ENV: &str = "ANANKE_EDDN_RELAY";
pub const EDDN_DISABLE_ENV: &str = "ANANKE_EDDN_DISABLE";
pub const EDDN_RECV_TIMEOUT_MS: i32 = 60_000;
pub const EDDN_RECONNECT_BASE_MS: u64 = 1_000;
pub const EDDN_RECONNECT_MAX_MS: u64 = 60_000;
pub const EDDN_FLUSH_INTERVAL_MS: u64 = 1_000;
pub const EDDN_FLUSH_BATCH_SIZE: usize = 200;

// --- Commander hotspot heatmap ---
pub const HEATMAP_X_MIN: f64 = -50_000.0;
pub const HEATMAP_X_MAX: f64 = 50_000.0;
pub const HEATMAP_Z_MIN: f64 = -25_000.0;
pub const HEATMAP_Z_MAX: f64 = 75_000.0;
pub const HEATMAP_W: usize = 1024;
pub const HEATMAP_H: usize = 1024;
pub const HEATMAP_DECAY_INTERVAL_SECS: u64 = 300;
pub const HEATMAP_DECAY_FACTOR: f64 = 0.9928057; // ≈8 hour half-life
pub const HEATMAP_RENDER_CACHE_SECS: u64 = 30;

// --- Routing ---
/// A* refinement budget. The whole chain has to fit inside Cloudflare's origin
/// response cap, because ananke.projectgaltea.org is proxied:
///
///   refine budget 80s  <  Caddy read_timeout 95s  <  Cloudflare 100s
///
/// Whichever link gives up first discards the greedy fallback result and hands
/// the caller an error instead, so the budget must be the tightest of the three.
pub const CARRIER_REFINE_BUDGET_MS: u128 = 80_000;
pub const CARRIER_JUMP_RANGE: f64 = 500.0;
pub const NEUTRON_REFINE_BUDGET_MS: u128 = 80_000;

// --- Admission control ---
/// Max concurrent heavy (A*/route-solve) requests. On the Deck (4c/8t), 2
/// means one carrier + one neutron can run simultaneously without thermal
/// throttling. Excess requests are rejected 503 immediately.
pub const ADMISSION_HEAVY: usize = 2;
/// Max concurrent lightweight requests (system lookups, cube search, EDMC
/// ingest, heatmap, etc). 128 is generous; it's really a safety valve to
/// stop a scrape flood from exhausting the tokio runtime.
pub const ADMISSION_LIGHT: usize = 128;

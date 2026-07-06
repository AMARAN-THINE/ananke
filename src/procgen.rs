/// ProcGen system name parser and coordinate estimator.
/// Ported from Esvandiary's EDTS (pgnames.py, pgdata.py, sector.py, util.py).
/// Derives approximate galactic coordinates from any procedurally generated
/// Elite Dangerous system name. Zero database dependency for sector lookup.

use std::sync::OnceLock;

const SECTOR_SIZE: f64 = 1280.0;
const GALAXY_SIZE: [usize; 3] = [128, 128, 128];
const BASE_SECTOR_INDEX: [i64; 3] = [39, 32, 18];
const BASE_COORDS: [f64; 3] = [-65.0, -25.0, -1065.0];

// ============================================================
// Fragment tables (from pgdata.py)
// ============================================================
const CX_PREFIXES: &[&str] = &[
  "Th","Eo","Oo","Eu","Tr","Sly","Dry","Ou","Tz","Phl","Ae","Sch","Hyp","Syst","Ai","Kyl",
  "Phr","Eae","Ph","Fl","Ao","Scr","Shr","Fly","Pl","Fr","Au","Pry","Pr","Hyph","Py","Chr",
  "Phyl","Tyr","Bl","Cry","Gl","Br","Gr","By","Aae","Myc","Gyr","Ly","Myl","Lych","Myn","Ch",
  "Myr","Cl","Rh","Wh","Pyr","Cr","Syn","Str","Syr","Cy","Wr","Hy","My","Sty","Sc","Sph",
  "Spl","A","Sh","B","C","D","Sk","Io","Dr","E","Sl","F","Sm","G","H","I",
  "Sp","J","Sq","K","L","Pyth","M","St","N","O","Ny","Lyr","P","Sw","Thr","Lys",
  "Q","R","S","T","Ea","U","V","W","Schr","X","Ee","Y","Z","Ei","Oe",
];

const C1_INFIXES_S1: &[&str] = &[
  "o","ai","a","oi","ea","ie","u","e","ee","oo","ue","i","oa","au","ae","oe",
];

const C1_INFIXES_S2: &[&str] = &[
  "ll","ss","b","c","d","f","dg","g","ng","h","j","k","l","m","n","mb",
  "p","q","gn","th","r","s","t","ch","tch","v","w","wh","ck","x","y","z","ph","sh","ct","wr",
];

const CX_SUFFIXES_S1: &[&str] = &[
  "oe","io","oea","oi","aa","ua","eia","ae","ooe","oo","a","ue","ai","e","iae","oae",
  "ou","uae","i","ao","au","o","eae","u","aea","ia","ie","eou","aei","ea","uia","oa","aae","eau","ee",
];

const C1_SUFFIXES_S2: &[&str] = &[
  "b","scs","wsy","c","d","vsky","f","sms","dst","g","rb","h","nts","ch","rd","rld",
  "k","lls","ck","rgh","l","rg","m","n","hm","p","hn","rk","q","rl","r","rm",
  "s","cs","wyg","rn","ct","t","hs","rbs","rp","tts","v","wn","ms","w","rr","mt",
  "x","rs","cy","y","rt","z","ws","lch","my","ry","nks","nd","sc","ng","sh","nk",
  "sk","nn","ds","sm","sp","ns","nt","dy","ss","st","rrs","xt","nz","sy","xy","rsch",
  "rphs","sts","sys","sty","th","tl","tls","rds","nch","rns","ts","wls","rnt","tt","rdy","rst",
  "pps","tz","tch","sks","ppy","ff","sps","kh","sky","ph","lts","wnst","rth","ths","fs","pp",
  "ft","ks","pr","ps","pt","fy","rts","ky","rshch","mly","py","bb","nds","wry","zz","nns",
  "ld","lf","gh","lks","sly","lk","ll","rph","ln","bs","rsts","gs","ls","vvy","lt","rks",
  "qs","rps","gy","wns","lz","nth","phs",
];

// All raw fragments in original order for greedy parsing
const CX_RAW_FRAGMENTS: &[&str] = &[
  "Th","Eo","Oo","Eu","Tr","Sly","Dry","Ou","Tz","Phl","Ae","Sch","Hyp","Syst","Ai","Kyl",
  "Phr","Eae","Ph","Fl","Ao","Scr","Shr","Fly","Pl","Fr","Au","Pry","Pr","Hyph","Py","Chr",
  "Phyl","Tyr","Bl","Cry","Gl","Br","Gr","By","Aae","Myc","Gyr","Ly","Myl","Lych","Myn","Ch",
  "Myr","Cl","Rh","Wh","Pyr","Cr","Syn","Str","Syr","Cy","Wr","Hy","My","Sty","Sc","Sph",
  "Spl","A","Sh","B","C","D","Sk","Io","Dr","E","Sl","F","Sm","G","H","I",
  "Sp","J","Sq","K","L","Pyth","M","St","N","O","Ny","Lyr","P","Sw","Thr","Lys",
  "Q","R","S","T","Ea","U","V","W","Schr","X","Ee","Y","Z","Ei","Oe",
  "ll","ss","b","c","d","f","dg","g","ng","h","j","k","l","m","n",
  "mb","p","q","gn","th","r","s","t","ch","tch","v","w","wh",
  "ck","x","y","z","ph","sh","ct","wr","o","ai","a","oi","ea",
  "ie","u","e","ee","oo","ue","i","oa","au","ae","oe","scs",
  "wsy","vsky","sms","dst","rb","nts","rd","rld","lls","rgh",
  "rg","hm","hn","rk","rl","rm","cs","wyg","rn","hs","rbs","rp",
  "tts","wn","ms","rr","mt","rs","cy","rt","ws","lch","my","ry",
  "nks","nd","sc","nk","sk","nn","ds","sm","sp","ns","nt","dy",
  "st","rrs","xt","nz","sy","xy","rsch","rphs","sts","sys","sty",
  "tl","tls","rds","nch","rns","ts","wls","rnt","tt","rdy","rst",
  "pps","tz","sks","ppy","ff","sps","kh","sky","lts","wnst","rth",
  "ths","fs","pp","ft","ks","pr","ps","pt","fy","rts","ky",
  "rshch","mly","py","bb","nds","wry","zz","nns","ld","lf",
  "gh","lks","sly","lk","rph","ln","bs","rsts","gs","ls","vvy",
  "lt","rks","qs","rps","gy","wns","lz","nth","phs","io","oea",
  "aa","ua","eia","ooe","iae","oae","ou","uae","ao","eae",
  "aea","ia","eou","aei","uia","aae","eau",
];

// ============================================================
// Override maps
// ============================================================
fn prefix_run_length(p: &str) -> usize {
    match p {
        "Eu"=>31,"Sly"=>4,"Tz"=>1,"Phl"=>13,"Ae"=>12,"Hyp"=>25,"Kyl"=>30,"Phr"=>10,
        "Eae"=>4,"Ao"=>5,"Scr"=>24,"Shr"=>11,"Fly"=>20,"Pry"=>3,"Hyph"=>14,"Py"=>12,
        "Phyl"=>8,"Tyr"=>25,"Cry"=>5,"Aae"=>5,"Myc"=>2,"Gyr"=>10,"Myl"=>12,"Lych"=>3,
        "Myn"=>10,"Myr"=>4,"Rh"=>15,"Wr"=>31,"Sty"=>4,"Spl"=>16,"Sk"=>27,"Sq"=>7,
        "Pyth"=>1,"Lyr"=>10,"Sw"=>24,"Thr"=>32,"Lys"=>10,"Schr"=>3,"Z"=>34,
        _ => 35,
    }
}

fn c2_suffix_idx(prefix: &str) -> usize {
    match prefix {
        "Eo"|"Oo"|"Eu"|"Ou"|"Ae"|"Ai"|"Eae"|"Ao"|"Au"|"Aae" => 2, _ => 1,
    }
}

fn c1_infix_idx(prefix: &str) -> usize {
    match prefix {
        "Eo"|"Oo"|"Eu"|"Ou"|"Ae"|"Ai"|"Eae"|"Ao"|"Au"|"Aae"
        |"A"|"Io"|"E"|"I"|"O"|"Ea"|"U"|"Ee"|"Ei"|"Oe" => 2, _ => 1,
    }
}

fn c1_infix_run_length(frag: &str) -> usize {
    match frag {
        "oi"=>88,"ue"=>147,"oa"=>57,"au"=>119,"ae"=>12,"oe"=>39,
        "dg"=>31,"tch"=>20,"wr"=>31,
        _ => if C1_INFIXES_S1.contains(&frag) { C1_SUFFIXES_S2.len() } else { CX_SUFFIXES_S1.len() }
    }
}

// ============================================================
// Precomputed lookup data
// ============================================================
struct PgLookup {
    fragments_sorted: Vec<String>,
    prefix_offsets: Vec<(usize, usize)>,
    prefix_total_run: usize,
    c1_infix_offsets_s1: Vec<(usize, usize)>,
    c1_infix_offsets_s2: Vec<(usize, usize)>,
    c1_infix_s1_total: usize,
    c1_infix_s2_total: usize,
}

fn build_lookup() -> PgLookup {
    let mut frags: Vec<String> = CX_RAW_FRAGMENTS.iter().map(|s| s.to_string()).collect();
    frags.sort_by(|a, b| b.len().cmp(&a.len()));
    frags.dedup();

    let mut prefix_offsets = Vec::with_capacity(CX_PREFIXES.len());
    let mut cnt = 0usize;
    for &p in CX_PREFIXES { let plen = prefix_run_length(p); prefix_offsets.push((cnt, plen)); cnt += plen; }
    let prefix_total_run = cnt;

    let mut c1_infix_offsets_s1 = Vec::with_capacity(C1_INFIXES_S1.len());
    cnt = 0;
    for &i in C1_INFIXES_S1 { let ilen = c1_infix_run_length(i); c1_infix_offsets_s1.push((cnt, ilen)); cnt += ilen; }
    let c1_infix_s1_total = cnt;

    let mut c1_infix_offsets_s2 = Vec::with_capacity(C1_INFIXES_S2.len());
    cnt = 0;
    for &i in C1_INFIXES_S2 { let ilen = c1_infix_run_length(i); c1_infix_offsets_s2.push((cnt, ilen)); cnt += ilen; }
    let c1_infix_s2_total = cnt;

    PgLookup { fragments_sorted: frags, prefix_offsets, prefix_total_run,
               c1_infix_offsets_s1, c1_infix_offsets_s2, c1_infix_s1_total, c1_infix_s2_total }
}

fn lookup() -> &'static PgLookup {
    static INSTANCE: OnceLock<PgLookup> = OnceLock::new();
    INSTANCE.get_or_init(build_lookup)
}

// ============================================================
// Utility (from util.py)
// ============================================================
fn jenkins32(mut key: u32) -> u32 {
    key = key.wrapping_add(key << 12); key ^= key >> 22;
    key = key.wrapping_add(key << 4);  key ^= key >> 9;
    key = key.wrapping_add(key << 10); key ^= key >> 2;
    key = key.wrapping_add(key << 7);  key ^= key >> 12;
    key
}

fn interleave(val1: u64, val2: u64, maxbits: u32) -> u64 {
    let mut output: u64 = 0;
    for i in 0..=(maxbits / 2) { output |= ((val1 >> i) & 1) << (i * 2); }
    for i in 0..=(maxbits / 2) { output |= ((val2 >> i) & 1) << (i * 2 + 1); }
    output & ((1u64 << maxbits) - 1)
}

// ============================================================
// Fragment parsing
// ============================================================
fn to_title_case(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut cap = true;
    for ch in s.chars() {
        if ch == ' ' || ch == '-' { cap = true; result.push(ch); }
        else if cap { for c in ch.to_uppercase() { result.push(c); } cap = false; }
        else { for c in ch.to_lowercase() { result.push(c); } }
    }
    result
}

fn get_sector_fragments(sector_name: &str) -> Option<Vec<String>> {
    let lu = lookup();
    let titled = to_title_case(sector_name).replace(' ', "");
    let mut remaining = titled.as_str();
    let mut segments = Vec::new();
    while !remaining.is_empty() {
        let mut found = false;
        for frag in &lu.fragments_sorted {
            if remaining.starts_with(frag.as_str()) {
                segments.push(frag.clone());
                remaining = &remaining[frag.len()..];
                found = true;
                break;
            }
        }
        if !found { return None; }
    }
    if segments.len() <= 4 { Some(segments) } else { None }
}

fn is_prefix(s: &str) -> bool { CX_PREFIXES.contains(&s) }

fn get_sector_class(frags: &[String]) -> Option<u8> {
    if frags.len() == 4 && is_prefix(&frags[0]) && is_prefix(&frags[2]) { Some(2) }
    else if (frags.len() == 3 || frags.len() == 4) && is_prefix(&frags[0]) { Some(1) }
    else { None }
}

// ============================================================
// Suffix/infix helpers
// ============================================================
fn get_suffixes_for_prefix(word_start: &str, get_all: bool) -> &'static [&'static str] {
    let idx = c2_suffix_idx(word_start);
    let full: &[&str] = if idx == 2 { &C1_SUFFIXES_S2[..CX_SUFFIXES_S1.len()] } else { CX_SUFFIXES_S1 };
    if get_all { full } else { &full[..prefix_run_length(word_start).min(full.len())] }
}

fn get_c1_suffixes(frags: &[String], get_all: bool) -> &'static [&'static str] {
    let last = frags.last().unwrap();
    if is_prefix(last) { return get_suffixes_for_prefix(last, get_all); }
    if C1_INFIXES_S2.contains(&last.as_str()) {
        if get_all { CX_SUFFIXES_S1 } else { &CX_SUFFIXES_S1[..prefix_run_length(&frags[0]).min(CX_SUFFIXES_S1.len())] }
    } else {
        let len = if get_all { C1_SUFFIXES_S2.len() } else { prefix_run_length(&frags[0]).min(C1_SUFFIXES_S2.len()) };
        &C1_SUFFIXES_S2[..len]
    }
}

fn c1_infix_offset(frag: &str) -> (usize, usize) {
    let lu = lookup();
    for (i, &inf) in C1_INFIXES_S1.iter().enumerate() { if inf == frag { return lu.c1_infix_offsets_s1[i]; } }
    for (i, &inf) in C1_INFIXES_S2.iter().enumerate() { if inf == frag { return lu.c1_infix_offsets_s2[i]; } }
    (0, 0)
}

fn c1_infix_total_run(frag: &str) -> usize {
    let lu = lookup();
    if C1_INFIXES_S1.contains(&frag) { lu.c1_infix_s1_total } else { lu.c1_infix_s2_total }
}

fn prefix_offset(p: &str) -> (usize, usize) {
    let lu = lookup();
    for (i, &px) in CX_PREFIXES.iter().enumerate() { if px == p { return lu.prefix_offsets[i]; } }
    (0, 0)
}

// ============================================================
// Class 2 offset (two-word names like "Flyua Phio")
// ============================================================
fn c2_get_offset_from_name(frags: &[String]) -> Option<u64> {
    if frags.len() != 4 { return None; }
    let sufs0 = get_suffixes_for_prefix(&frags[0], false);
    let sufs1 = get_suffixes_for_prefix(&frags[2], false);
    let idx0 = sufs0.iter().position(|&s| s == frags[1])? + prefix_offset(&frags[0]).0;
    let idx1 = sufs1.iter().position(|&s| s == frags[3])? + prefix_offset(&frags[2]).0;
    Some(interleave(idx0 as u64, idx1 as u64, 32))
}

// ============================================================
// Class 1 offset (one-word names like "Wregoe")
// ============================================================
fn c1_get_offset_from_name(frags: &[String]) -> Option<u64> {
    let sufs = get_c1_suffixes(&frags[..frags.len()-1], true);
    let suf_offset_raw = sufs.iter().position(|&s| s == frags[frags.len()-1])?;
    let mut f3_offset = suf_offset_raw;

    if frags.len() > 3 {
        let f3_run = c1_infix_run_length(&frags[2]);
        let f3_total = c1_infix_total_run(&frags[2]);
        let adjusted = suf_offset_raw + (suf_offset_raw / f3_run) * f3_total;
        let (q, r) = (adjusted / f3_run, adjusted % f3_run);
        f3_offset = q * f3_total + r + c1_infix_offset(&frags[2]).0;
    }

    let f2_run = c1_infix_run_length(&frags[1]);
    let f2_total = c1_infix_total_run(&frags[1]);
    let (q2, r2) = (f3_offset / f2_run, f3_offset % f2_run);
    let f2_offset = q2 * f2_total + r2 + c1_infix_offset(&frags[1]).0;

    let lu = lookup();
    let p_run = prefix_run_length(&frags[0]);
    let (q3, r3) = (f2_offset / p_run, f2_offset % p_run);
    let offset = q3 * lu.prefix_total_run + r3 + prefix_offset(&frags[0]).0;

    Some(offset as u64)
}

// ============================================================
// Offset to sector coordinates
// ============================================================
fn sector_pos_from_offset(offset: u64) -> (i64, i64, i64) {
    let o = offset as i64;
    let gx = GALAXY_SIZE[0] as i64;
    let gy = GALAXY_SIZE[1] as i64;
    (o % gx - BASE_SECTOR_INDEX[0], (o / gx) % gy - BASE_SECTOR_INDEX[1], o / (gx * gy) - BASE_SECTOR_INDEX[2])
}

/// Sector lookup: name -> relative sector coordinates. Pure math, no DB.
pub fn sector_from_name(sector_name: &str) -> Option<(i64, i64, i64)> {
    let frags = get_sector_fragments(sector_name)?;
    let sc = get_sector_class(&frags)?;
    let offset = match sc {
        2 => c2_get_offset_from_name(&frags)?,
        1 => {
            let raw = c1_get_offset_from_name(&frags)?;
            if (jenkins32(raw as u32) % 2) + 1 != 1 { return None; }
            raw
        }
        _ => return None,
    };
    Some(sector_pos_from_offset(offset))
}

// ============================================================
// Public API
// ============================================================
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ProcGenName {
    pub sector_name: String,
    pub l1: u32, pub l2: u32, pub l3: u32,
    pub mass_code: u32, pub mass_char: char,
    pub n1: u32, pub n2: u32,
}

#[derive(Debug, Clone)]
pub struct EstimatedCoords {
    pub x: f64, pub y: f64, pub z: f64,
    pub uncertainty_ly: f64,
}

pub fn parse_procgen_name(name: &str) -> Option<ProcGenName> {
    let name = name.trim();
    if name.len() < 10 { return None; }
    let last_space = name.rfind(' ')?;
    let tail = &name[last_space + 1..];
    let prefix = &name[..last_space];
    let mass_char = tail.chars().next()?;
    if !mass_char.is_ascii_lowercase() || mass_char < 'a' || mass_char > 'h' { return None; }
    let mass_code = (mass_char as u32) - ('a' as u32);
    let numbers = &tail[1..];
    let dash_pos = numbers.find('-')?;
    let n1: u32 = numbers[..dash_pos].parse().ok()?;
    let n2: u32 = numbers[dash_pos + 1..].parse().ok()?;
    let prefix = prefix.trim();
    if prefix.len() < 4 { return None; }
    let last_space2 = prefix.rfind(' ')?;
    let subsector = &prefix[last_space2 + 1..];
    let sector_name = prefix[..last_space2].trim();
    if subsector.len() != 4 { return None; }
    let sb = subsector.as_bytes();
    if !sb[0].is_ascii_uppercase() || !sb[1].is_ascii_uppercase() || sb[2] != b'-' || !sb[3].is_ascii_uppercase() { return None; }
    if sector_name.is_empty() || !sector_name.as_bytes()[0].is_ascii_uppercase() { return None; }
    Some(ProcGenName {
        sector_name: sector_name.to_string(),
        l1: (sb[0]-b'A') as u32, l2: (sb[1]-b'A') as u32, l3: (sb[3]-b'A') as u32,
        mass_code, mass_char, n1, n2,
    })
}

pub fn estimate_coords(pg: &ProcGenName) -> Option<EstimatedCoords> {
    let (sx, sy, sz) = sector_from_name(&pg.sector_name)?;
    let cubeside = 10.0 * (1u32 << pg.mass_code) as f64;
    let bid = pg.n1 * 17576 + pg.l3 * 676 + pg.l2 * 26 + pg.l1;
    let column = bid % 128;
    let stack  = (bid / 128) % 128;
    let row    = bid / (128 * 128);
    let half = cubeside / 2.0;
    Some(EstimatedCoords {
        x: BASE_COORDS[0] + sx as f64 * SECTOR_SIZE + column as f64 * cubeside + half,
        y: BASE_COORDS[1] + sy as f64 * SECTOR_SIZE + stack  as f64 * cubeside + half,
        z: BASE_COORDS[2] + sz as f64 * SECTOR_SIZE + row    as f64 * cubeside + half,
        uncertainty_ly: half,
    })
}

pub fn sector_coords_from_id64(id64: i64) -> (u32, u32, u32) {
    let id = id64 as u64;
    let mc = id & 7;
    let bpe = 128u64 >> mc;
    let raw_x = ((id >> (30 - mc * 2)) & (0x3FFF >> mc)) as u32;
    let raw_y = ((id >> (17 - mc))      & (0x1FFF >> mc)) as u32;
    let raw_z = ((id >> 3)              & (0x3FFF >> mc)) as u32;
    (raw_x / bpe as u32, raw_y / bpe as u32, raw_z / bpe as u32)
}

/// Resolve a system name or id64 string to (id64, name, x, y, z).
/// Tries the DB first, then falls back to ProcGen coordinate estimation.
/// For estimated systems, id64 is set to -1 (synthetic).
pub fn resolve_system(conn: &rusqlite::Connection, input: &str) -> Result<(i64, String, f64, f64, f64), String> {
    // Try as id64 first
    if let Ok(id) = input.parse::<i64>() {
        if let Ok(row) = conn.query_row(
            "SELECT s.id64, s.name, i.minX, i.minY, i.minZ \
             FROM systems s JOIN systems_index i ON s.id64=i.id \
             WHERE s.id64=? LIMIT 1",
            rusqlite::params![id],
            |r| Ok((r.get::<_,i64>(0)?, r.get::<_,String>(1)?, r.get::<_,f64>(2)?, r.get::<_,f64>(3)?, r.get::<_,f64>(4)?)),
        ) {
            return Ok(row);
        }
    }
    // Try as name
    if let Ok(row) = conn.query_row(
        "SELECT s.id64, s.name, i.minX, i.minY, i.minZ \
         FROM systems s JOIN systems_index i ON s.id64=i.id \
         WHERE s.name=? COLLATE NOCASE LIMIT 1",
        rusqlite::params![input],
        |r| Ok((r.get::<_,i64>(0)?, r.get::<_,String>(1)?, r.get::<_,f64>(2)?, r.get::<_,f64>(3)?, r.get::<_,f64>(4)?)),
    ) {
        return Ok(row);
    }
    // Fallback: ProcGen coordinate estimation
    if let Some(pg) = parse_procgen_name(input) {
        if let Some(est) = estimate_coords(&pg) {
            return Ok((-1, input.to_string(), est.x, est.y, est.z));
        }
    }
    Err(format!("System '{}' not found", input))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_colonia() {
        let pg = parse_procgen_name("Eol Prou RS-T d3-94").unwrap();
        assert_eq!(pg.sector_name, "Eol Prou");
        assert_eq!((pg.l1, pg.l2, pg.l3), (17, 18, 19));
        assert_eq!(pg.mass_code, 3);
    }

    #[test]
    fn sector_colonia() {
        let s = sector_from_name("Eol Prou").unwrap();
        assert_eq!(s, (-8, -1, 16));
    }

    #[test]
    fn coords_colonia() {
        let pg = parse_procgen_name("Eol Prou RS-T d3-94").unwrap();
        let est = estimate_coords(&pg).unwrap();
        assert!((est.x - (-9530.5)).abs() < 80.0, "x={}", est.x);
        assert!((est.y - (-910.28)).abs() < 80.0, "y={}", est.y);
        assert!((est.z - 19808.12).abs() < 80.0, "z={}", est.z);
    }

    #[test]
    fn sector_wregoe() {
        let s = sector_from_name("Wregoe").unwrap();
        assert_eq!(s, (0, 0, 0));
    }

    #[test]
    fn rejects_non_procgen() {
        assert!(parse_procgen_name("Sol").is_none());
        assert!(parse_procgen_name("Colonia").is_none());
    }
}

use h3o::CellIndex;
use serde::Deserialize;

pub fn serialize_neighbor_cells<S>(cells: &Vec<CellIndex>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use serde::ser::SerializeSeq;
    let mut seq = serializer.serialize_seq(Some(cells.len()))?;
    for cell in cells {
        let value: u64 = (*cell).into();
        seq.serialize_element(&value)?;
    }
    seq.end()
}

pub fn deserialize_neighbor_cells<'de, D>(deserializer: D) -> Result<Vec<CellIndex>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw_cells = Vec::<u64>::deserialize(deserializer)?;
    raw_cells
        .into_iter()
        .map(|raw| CellIndex::try_from(raw).map_err(serde::de::Error::custom))
        .collect()
}

pub fn serialize_cell_index<S>(cell: &CellIndex, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let value: u64 = (*cell).into();
    serializer.serialize_u64(value)
}

pub fn deserialize_cell_index<'de, D>(deserializer: D) -> Result<CellIndex, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw = u64::deserialize(deserializer)?;
    CellIndex::try_from(raw).map_err(serde::de::Error::custom)
}

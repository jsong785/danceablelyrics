mod query;
use anyhow::Result;
use clap::{Args, Parser};
use std::path::PathBuf;

use mimalloc::MiMalloc;
use polars::prelude::{col, CsvWriter};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Parser, Debug)]
struct Configuration {
    #[command(flatten)]
    files: Files,
    #[arg(long)]
    keywords: Vec<String>,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 1000)]
    min_views: i32,
}

#[derive(Args, Debug)]
struct Files {
    #[arg(long)]
    artists: PathBuf,
    #[arg(long)]
    audio_features: PathBuf,
    #[arg(long)]
    genius_song_lyrics: PathBuf,
    #[arg(long)]
    r_track_artist: PathBuf,
    #[arg(long)]
    tracks: PathBuf,
}

fn main() -> Result<()> {
    println!("parsing arguments...");
    let config = Configuration::try_parse()?;

    println!("building genius song lyrics query...");
    let genius_song_lyrics = query::genius_song_lyrics(
        &config.files.genius_song_lyrics,
        config.keywords.as_slice(),
        config.min_views,
    )?;

    println!("building danceable track information query...");
    let danceable_tracks = query::danceable_tracks(
        &config.files.artists,
        &config.files.tracks,
        &config.files.audio_features,
        &config.files.r_track_artist,
    )?;

    println!("building result...");
    use polars::prelude::SortOptions;
    let mut matches = danceable_tracks
        .join(
            genius_song_lyrics,
            [col("artist_name"), col("track_name")],
            [col("artist_name"), col("track_name")],
            polars::prelude::JoinType::Inner,
        )
        .sort(
            "danceability",
            SortOptions {
                descending: true,
                ..Default::default()
            },
        )
        .collect()?;

    println!("writing to file...");
    let output_file = std::fs::File::create(config.output)?;
    use polars::prelude::SerWriter;
    let mut writer = CsvWriter::new(output_file);
    writer.finish(&mut matches)?;
    Ok(())
}

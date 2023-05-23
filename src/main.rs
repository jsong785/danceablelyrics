use anyhow::Result;
use clap::{Args, Parser};
use std::path::PathBuf;

use mimalloc::MiMalloc;
use polars::{
    lazy::dsl::Expr,
    prelude::{col, lit, CsvWriter, LazyCsvReader},
};

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
    let genius_song_lyrics = {
        use polars::prelude::LazyFileListReader;
        let csv = LazyCsvReader::new(&config.files.genius_song_lyrics)
            .has_header(true)
            .finish()?;
        let csv = csv.select(&[
            col("title").str().to_lowercase().alias("track_name"),
            col("artist").str().to_lowercase().alias("artist_name"),
            "lyrics".into(),
            "language".into(),
            "views".into(),
        ]);
        let csv = csv.filter(
            col("views").gt(lit(config.min_views))
                + col("language").eq(lit("en"))
                + (|| -> Expr {
                    let contains_lyrics = config
                        .keywords
                        .iter()
                        .map(|c| col("lyrics").str().contains_literal(lit(c.as_str())));
                    let res = contains_lyrics.fold(Expr::default(), |accum, item| accum * item);
                    res
                }()),
        );
        csv.select(&["track_name".into(), "artist_name".into()])
    };

    println!("building danceable track information query...");
    let danceable_tracks = {
        let artists = {
            use polars::prelude::LazyFileListReader;
            let csv = LazyCsvReader::new(&config.files.artists)
                .has_header(true)
                .finish()?;
            let csv = csv.select(&[
                col("name").str().to_lowercase().alias("artist_name"),
                col("id").alias("artist_id"),
            ]);
            csv
        };

        let tracks = {
            use polars::prelude::LazyFileListReader;
            let csv = LazyCsvReader::new(&config.files.tracks)
                .has_header(true)
                .finish()?;
            let csv = csv.select(&[
                col("name").str().to_lowercase().alias("track_name"),
                col("id").alias("track_id"),
                "explicit".into(),
            ]);
            csv.filter(col("explicit").eq(0))
        };

        let audio_features = {
            use polars::prelude::LazyFileListReader;
            let csv = LazyCsvReader::new(&config.files.audio_features)
                .has_header(true)
                .finish()?;
            let csv = csv.select(&[
                col("id").alias("track_id"),
                "danceability".into(),
                "energy".into(),
                "tempo".into(),
            ]);
            const DANCEABILITY: (f64, f64) = (0.45, 0.99);
            const ENERGY: (f32, f32) = (0.45, 0.75);
            const TEMPO: (f32, f32) = (110.0, 140.0);
            csv.filter(
                col("danceability").gt_eq(DANCEABILITY.0)
                    * col("danceability").lt_eq(DANCEABILITY.1)
                    * col("energy").gt_eq(ENERGY.0)
                    * col("energy").lt_eq(ENERGY.1)
                    * col("tempo").gt_eq(TEMPO.0)
                    * col("tempo").lt_eq(TEMPO.1),
            )
        };

        let track_artists = {
            use polars::prelude::LazyFileListReader;
            let csv = LazyCsvReader::new(&config.files.r_track_artist)
                .has_header(true)
                .finish()?;
            let csv = csv.select(&["track_id".into(), "artist_id".into()]);
            let csv = csv.inner_join(artists, col("artist_id"), col("artist_id"));
            let csv = csv.inner_join(tracks, col("track_id"), col("track_id"));
            let csv = csv.inner_join(audio_features, col("track_id"), col("track_id"));
            csv
        };

        let danceability_tracks = {
            let ta = track_artists.select(&[
                "artist_name".into(),
                "track_name".into(),
                "danceability".into(),
                "energy".into(),
                lit("https://open.spotify.com/track/") + col("track_id"),
            ]);
            use polars::lazy::dsl::all;
            use polars::lazy::prelude::*;
            let other = ta.groupby([col("artist_name"), col("track_name")]);
            other.agg([all().sort_by([col("danceability")], [true]).first()])
        };
        danceability_tracks
    };

    use polars::prelude::SortOptions;
    let mut matches = danceable_tracks
        .inner_join(
            genius_song_lyrics,
            col("artist_name") + col("track_name"),
            col("artist_name") + col("track_name"),
        )
        .sort(
            "danceability",
            SortOptions {
                descending: true,
                ..Default::default()
            },
        )
        .collect()?;

    let output_file = std::fs::File::create(config.output)?;
    use polars::prelude::SerWriter;
    let mut writer = CsvWriter::new(output_file);
    writer.finish(&mut matches)?;
    Ok(())
}

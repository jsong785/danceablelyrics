use anyhow::Result;
use polars::prelude::{col, lit, Expr, LazyCsvReader, LazyFileListReader, LazyFrame};
use std::path::PathBuf;

pub fn genius_song_lyrics(
    file: &PathBuf,
    keywords: &[String],
    min_views: i32,
) -> Result<LazyFrame> {
    let csv = LazyCsvReader::new(file).has_header(true).finish()?;
    let csv = csv.select(&[
        col("title").alias("track_name"),
        col("artist").alias("artist_name"),
        "lyrics".into(),
        "language".into(),
        "views".into(),
    ]);
    let csv = csv.filter(
        col("views")
            .gt(lit(min_views))
            .and(col("language").eq(lit("en")))
            .and(|| -> Expr {
                let contains_lyrics = keywords.iter().map(|c| c.to_lowercase()).map(|c| {
                    col("lyrics")
                        .str()
                        .to_lowercase()
                        .str()
                        .contains_literal(lit(c.as_str()))
                });
                contains_lyrics.fold(Expr::default(), |accum, item| accum.or(item))
            }()),
    );
    let csv = csv.select(&[
        col("track_name").str().to_lowercase(),
        col("artist_name").str().to_lowercase(),
    ]);
    Ok(csv)
}

pub fn danceable_tracks(
    artists: &PathBuf,
    tracks: &PathBuf,
    audio_features: &PathBuf,
    r_track_artist: &PathBuf,
) -> Result<LazyFrame> {
    let artists = {
        let csv = LazyCsvReader::new(artists).has_header(true).finish()?;
        csv.select(&[
            col("name").str().to_lowercase().alias("artist_name"),
            col("id").alias("artist_id"),
        ])
    };

    let tracks = {
        let csv = LazyCsvReader::new(tracks).has_header(true).finish()?;
        let csv = csv.select(&[
            col("name").str().to_lowercase().alias("track_name"),
            col("id").alias("track_id"),
            "explicit".into(),
        ]);
        csv.filter(col("explicit").eq(0))
    };

    let audio_features = {
        let csv = LazyCsvReader::new(audio_features)
            .has_header(true)
            .finish()?;
        let csv = csv.select(&[
            col("id").alias("track_id"),
            "danceability".into(),
            "energy".into(),
            "tempo".into(),
        ]);
        const DANCEABILITY: (f32, f32) = (0.45, 0.99);
        const ENERGY: (f32, f32) = (0.45, 0.75);
        const TEMPO: (f32, f32) = (110.0, 140.0);
        csv.filter(
            col("danceability")
                .gt_eq(DANCEABILITY.0)
                .and(col("danceability").lt_eq(DANCEABILITY.1))
                .and(col("energy").gt_eq(ENERGY.0))
                .and(col("energy").lt_eq(ENERGY.1))
                .and(col("tempo").gt_eq(TEMPO.0))
                .and(col("tempo").lt_eq(TEMPO.1)),
        )
    };

    let track_artists = {
        let csv = LazyCsvReader::new(r_track_artist)
            .has_header(true)
            .finish()?;
        let csv = csv.select(&["track_id".into(), "artist_id".into()]);
        let csv = csv.inner_join(artists, col("artist_id"), col("artist_id"));
        let csv = csv.inner_join(tracks, col("track_id"), col("track_id"));
        csv.inner_join(audio_features, col("track_id"), col("track_id"))
    };

    let danceability = track_artists.select(&[
        "artist_name".into(),
        "track_name".into(),
        "danceability".into(),
        "energy".into(),
        (lit("https://open.spotify.com/track/") + col("track_id")).alias("track_id"),
    ]);
    use polars::lazy::dsl::all;
    use polars::lazy::prelude::*;
    let danceability = danceability.groupby([col("artist_name"), col("track_name")]);
    let danceability = danceability.agg([all().sort_by([col("danceability")], [true]).first()]);
    Ok(danceability)
}

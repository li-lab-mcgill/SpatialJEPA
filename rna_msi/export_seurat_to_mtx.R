#!/usr/bin/env Rscript
# Export aligned se.multi.list to Matrix Market + metadata for Python/MuData.
#
# Usage (murine samples, default):
#   Rscript export_seurat_to_mtx.R
#
# Usage (human V11T17-102 sections from nearest_neighbors CSVs):
#   Rscript export_seurat_to_mtx.R \
#     --input-rds hPDStr.multi.list \
#     --from-neighbors-csv "/path/to/results/tables/V11T17-102_*_nearest_neighbors.csv"
#
#   $PY_MSI build_mudata_from_mtx.py

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
})

parse_args <- function(args) {
  opts <- list(
    input_rds = "se.multi.list",
    samples = NULL,
    from_neighbors_csv = NULL,
    knn_rds = "knn_spatial_df_filtered_list"
  )
  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]
    if (key %in% c("--input-rds", "--samples", "--from-neighbors-csv", "--knn-rds")) {
      if (i == length(args)) stop("Missing value for ", key)
      value <- args[[i + 1L]]
      switch(
        key,
        "--input-rds" = { opts$input_rds <- value },
        "--samples" = { opts$samples <- strsplit(value, ",", fixed = TRUE)[[1]] },
        "--from-neighbors-csv" = { opts$from_neighbors_csv <- value },
        "--knn-rds" = { opts$knn_rds <- value }
      )
      i <- i + 2L
    } else {
      stop("Unknown argument: ", key)
    }
  }
  opts
}

sample_id_from_neighbors_csv <- function(path) {
  basename <- tools::file_path_sans_ext(basename(path))
  sub("_nearest_neighbors$", "", basename)
}

load_alignment_from_neighbors_csv <- function(path) {
  alignment <- read.csv(path, stringsAsFactors = FALSE)
  required_cols <- c("from", "to", "distance", "x", "x_end", "y", "y_end")
  missing_cols <- setdiff(required_cols, colnames(alignment))
  if (length(missing_cols) > 0) {
    stop(path, " is missing columns: ", paste(missing_cols, collapse = ", "))
  }
  alignment[, required_cols, drop = FALSE]
}

working.dir <- "${SPATIALJEPA_HOME}/outputs/SMA"
r_dir <- file.path(working.dir, "R_objects")
out_dir <- file.path(working.dir, "h5mu_export")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cli <- parse_args(commandArgs(trailingOnly = TRUE))

se.multi.list <- readRDS(file.path(r_dir, cli$input_rds))
alignment_by_sample <- list()

if (!is.null(cli$from_neighbors_csv)) {
  neighbor_paths <- Sys.glob(cli$from_neighbors_csv)
  if (length(neighbor_paths) == 0) {
    stop("No files matched --from-neighbors-csv: ", cli$from_neighbors_csv)
  }
  for (path in neighbor_paths) {
    sample_id <- sample_id_from_neighbors_csv(path)
    alignment_by_sample[[sample_id]] <- load_alignment_from_neighbors_csv(path)
  }
  if (is.null(cli$samples)) {
    cli$samples <- names(alignment_by_sample)
  }
} else {
  knn_path <- file.path(r_dir, cli$knn_rds)
  knn_spatial_df_filtered_list <- if (file.exists(knn_path)) readRDS(knn_path) else NULL

  if (is.null(names(se.multi.list)) || any(names(se.multi.list) == "")) {
    se.rna.list <- readRDS(file.path(r_dir, "se.RNA.list"))
    if (length(se.multi.list) != length(se.rna.list)) {
      stop("se.multi.list and se.RNA.list have different lengths; cannot assign sample names.")
    }
    names(se.multi.list) <- names(se.rna.list)
  }

  if (!is.null(knn_spatial_df_filtered_list)) {
    if (is.null(names(knn_spatial_df_filtered_list)) || any(names(knn_spatial_df_filtered_list) == "")) {
      if (length(knn_spatial_df_filtered_list) != length(se.multi.list)) {
        stop("knn_spatial_df_filtered_list and se.multi.list have different lengths; cannot assign sample names.")
      }
      names(knn_spatial_df_filtered_list) <- names(se.multi.list)
    }
    alignment_by_sample <- knn_spatial_df_filtered_list
  }
}

if (is.null(names(se.multi.list)) || any(names(se.multi.list) == "")) {
  stop("Sample names are missing from ", cli$input_rds, "; pass --samples explicitly.")
}

sample_ids <- if (is.null(cli$samples)) names(se.multi.list) else cli$samples
missing_samples <- setdiff(sample_ids, names(se.multi.list))
if (length(missing_samples) > 0) {
  stop("Samples not found in ", cli$input_rds, ": ", paste(missing_samples, collapse = ", "))
}

if (length(alignment_by_sample) > 0) {
  missing_alignment <- setdiff(sample_ids, names(alignment_by_sample))
  if (length(missing_alignment) > 0) {
    stop("Missing alignment metadata for samples: ", paste(missing_alignment, collapse = ", "))
  }
}

add_alignment_metadata <- function(obs, alignment, sample_id) {
  if (is.null(alignment)) {
    warning("No alignment metadata found for sample ", sample_id)
    return(obs)
  }

  alignment <- as.data.frame(alignment)
  required_cols <- c("from", "to", "distance", "x", "x_end", "y", "y_end")
  missing_cols <- setdiff(required_cols, colnames(alignment))
  if (length(missing_cols) > 0) {
    stop(
      "Alignment metadata for sample ", sample_id,
      " is missing columns: ", paste(missing_cols, collapse = ", ")
    )
  }

  alignment <- alignment[, required_cols, drop = FALSE]
  colnames(alignment) <- c(
    "msi_barcode", "barcode", "alignment_distance",
    "msi_warped_x", "rna_warped_x", "msi_warped_y", "rna_warped_y"
  )

  if (anyDuplicated(alignment$barcode)) {
    stop("Alignment metadata for sample ", sample_id, " contains duplicate RNA barcodes.")
  }

  obs <- merge(obs, alignment, by = "barcode", all.x = TRUE, sort = FALSE)
  missing_alignment <- sum(is.na(obs$msi_barcode))
  if (missing_alignment > 0) {
    warning(
      "Sample ", sample_id, " has ", missing_alignment,
      " observations without alignment metadata."
    )
  }

  obs
}

export_one <- function(se, sample_id, out_dir, alignment = NULL) {
  sample_dir <- file.path(out_dir, sample_id)
  dir.create(sample_dir, recursive = TRUE, showWarnings = FALSE)

  rna <- LayerData(se, assay = "RNA", layer = "counts")
  msi <- LayerData(se, assay = "MSI", layer = "counts")

  if (!all(colnames(rna) == colnames(msi))) {
    stop("RNA and MSI barcodes differ for sample ", sample_id)
  }

  obs <- se@meta.data
  obs$barcode <- rownames(obs)

  if ("Staffli" %in% names(se@tools)) {
    staffli <- se@tools$Staffli@meta.data
    staffli$barcode <- rownames(staffli)
    obs <- merge(obs, staffli, by = "barcode", all.x = TRUE, suffixes = c("", "_staffli"))
  }

  obs$sample_id <- sample_id
  obs <- add_alignment_metadata(obs, alignment, sample_id)

  # AnnData/MuData use cells x features.
  writeMM(t(rna), file.path(sample_dir, "rna.mtx"))
  writeMM(t(msi), file.path(sample_dir, "msi.mtx"))
  write.table(rownames(rna), file.path(sample_dir, "rna_features.tsv"),
              quote = FALSE, col.names = FALSE, row.names = FALSE)
  write.table(rownames(msi), file.path(sample_dir, "msi_features.tsv"),
              quote = FALSE, col.names = FALSE, row.names = FALSE)
  write.table(colnames(rna), file.path(sample_dir, "barcodes.tsv"),
              quote = FALSE, col.names = FALSE, row.names = FALSE)
  write.table(obs, file.path(sample_dir, "obs.tsv"),
              quote = FALSE, sep = "\t", row.names = FALSE)

  message("Wrote ", sample_dir, " (", ncol(se), " paired spots)")
}

for (sample_id in sample_ids) {
  alignment <- if (length(alignment_by_sample) > 0) {
    alignment_by_sample[[sample_id]]
  } else {
    NULL
  }
  export_one(se.multi.list[[sample_id]], sample_id, out_dir, alignment)
}

message("Done. Run: $PY_MSI build_mudata_from_mtx.py")

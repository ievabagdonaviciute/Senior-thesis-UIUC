# === CLEVRER: per-category PNGs (2 cols: Lasso, Elastic Net) ===
library(glmnet)
library(jsonlite)
library(tools)

# ----- Paths -----
setwd("/home/ievab2/run_models/full_8_frames_qwen_vs_internvl/QWEN/embeddings/only_prompt")
csv_path   <- "prompts_meta.csv"                  # row_idx,prompt,correct,raw_score
emb_path   <- "embeddings_eva02_l14.csv.gz"       # embeddings (rows align with CSV)
jsonl_path <- "/home/ievab2/run_models/questions/clevrer_filtered_500.jsonl"
out_dir    <- "cat_analysis_log_reg"

# Ensure output dir exists
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# ----- Load data -----
meta  <- read.csv(csv_path, stringsAsFactors = FALSE)
X     <- as.matrix(read.csv(gzfile(emb_path)))
y_raw <- meta$correct
y     <- if (any(!(unique(y_raw) %in% c(0,1)))) as.integer(y_raw >= 1.0) else as.integer(y_raw)
keep  <- !is.na(y)
meta  <- meta[keep, , drop = FALSE]; X <- X[keep, , drop = FALSE]; y <- y[keep]

# ----- Map categories via strict 0-based row_idx -> JSONL line -----
jlines   <- readLines(jsonl_path, warn = FALSE)
cats_all <- vapply(jlines, function(z) fromJSON(z)$category, character(1))
if (!"row_idx" %in% names(meta)) stop("CSV must contain 'row_idx' (0-based).")
meta$category <- cats_all[meta$row_idx + 1]

# ----- Helpers -----
add_prior_line <- function(prior_err) {
  usr <- par("usr")
  segments(usr[1], prior_err, usr[2], prior_err, col="blue", lty=2, lwd=2)
}

annotate_stats <- function(n, acc, prior) {
  txt <- sprintf("n=%d   acc≈%s   prior=%.3f", n, ifelse(is.na(acc), "NA", sprintf("%.3f", acc)), prior)
  mtext(txt, side = 3, adj = 1, line = -1, cex = 0.9, col = "gray30")
}

plot_panel <- function(Xs, ys, alpha_val, title_txt) {
  n <- length(ys)
  p1 <- mean(ys == 1)
  prior_error <- min(p1, 1 - p1)

  if (length(unique(ys)) < 2 || nrow(Xs) < 5) {
    # show prior only
    plot(0, 0, type="n", axes=FALSE, xlab="", ylab="", main="")
    box(); title(main = title_txt, cex.main = 1.05, line = 1.5)
    add_prior_line(prior_error)
    legend("bottomleft",
           legend = c("Dashed blue = Prior"),
           col = c("blue"), lty = c(2), lwd = c(2),
           bty = "n", cex = 0.9)
    annotate_stats(n = n, acc = NA_real_, prior = prior_error)
    return(list(n = n, acc = NA_real_, prior = prior_error))
  }

  set.seed(42)
  foldid <- sample(1:10, size = n, replace = TRUE)
  cvfit  <- cv.glmnet(Xs, ys, family="binomial", alpha=alpha_val,
                      foldid=foldid, type.measure="class")

  # cv.glmnet's default plot: CV error vs log(lambda)
  plot(cvfit, main = "", cex.axis = 0.95, cex.lab = 1.05)
  title(main = title_txt, line = 2.2, cex.main = 1.05)

  # Prior error (dashed blue)
  add_prior_line(prior_error)

  # Accuracy at lambda.min
  idx_min <- which.min(cvfit$cvm)
  acc_min <- 1 - cvfit$cvm[idx_min]
  annotate_stats(n = n, acc = acc_min, prior = prior_error)

  # Tiny legend to avoid separate rows
  legend("bottomleft",
         legend = c("CV error", "Prior"),
         col = c("red", "blue"), lty = c(1, 2), lwd = c(2, 2),
         bty = "n", cex = 0.9)
  list(n = n, acc = acc_min, prior = prior_error)
}

# ----- Categories -----
cats <- c("descriptive","predictive","explanatory","counterfactual")

# Collect stats for a summary table
all_stats <- list()

# ----- Save 4 separate PNGs (one per category) -----
for (cat in cats) {
  sel <- which(meta$category == cat)
  Xc  <- X[sel, , drop = FALSE]
  yc  <- y[sel]

  # Horizontal: 1 row × 2 cols
  png(file.path(out_dir, sprintf("cv_logreg_%s.png", cat)),
      width = 2400, height = 1200, res = 200)

  layout(matrix(1:2, nrow = 1, byrow = TRUE))
  par(mar = c(4, 4, 3.2, 1), mgp = c(2.4, 0.9, 0))

  # Left: Lasso (alpha=1)
  title_lasso <- sprintf("%s — Lasso (α=1)", toTitleCase(cat))
  s1 <- plot_panel(Xc, yc, alpha_val = 1, title_txt = title_lasso)
  all_stats[[paste0("lasso_", cat)]] <- c(category = cat, model="lasso", n=s1$n, acc=s1$acc, prior=s1$prior)

  # Right: Elastic Net (alpha=0.5)
  title_enet <- sprintf("%s — Elastic Net (α=0.5)", toTitleCase(cat))
  s2 <- plot_panel(Xc, yc, alpha_val = 0.5, title_txt = title_enet)
  all_stats[[paste0("enet_", cat)]]  <- c(category = cat, model="elastic_net", n=s2$n, acc=s2$acc, prior=s2$prior)

  dev.off()
  cat(sprintf("✅ Saved: %s\n", file.path(getwd(), out_dir, sprintf("cv_logreg_%s.png", cat))))
}

# ----- Compact summary table -----
summ <- do.call(rbind, lapply(all_stats, function(v) {
  data.frame(category = v[["category"]],
             model    = v[["model"]],
             n        = as.integer(v[["n"]]),
             acc      = as.numeric(v[["acc"]]),
             prior    = as.numeric(v[["prior"]]),
             row.names = NULL)
}))
print(summ[order(summ$category, summ$model), ], row.names = FALSE, digits = 4)

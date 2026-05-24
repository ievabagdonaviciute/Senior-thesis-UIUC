# === Prior vs Prediction: clean legend below plots (titles slightly smaller) ===

library(glmnet)
setwd("/home/ievab2/run_models/full_8_frames_qwen_vs_internvl/INTERNVL/embeddings/image_and_prompt")

# --- Load data ---
meta <- read.csv("image_and_prompt_meta.csv", stringsAsFactors = FALSE)
X <- as.matrix(read.csv(gzfile("embeddings_eva02_l14_both.csv.gz")))
y_raw <- meta$correct

# Convert to 0/1 (if partial credit present)
y <- if (any(!(unique(y_raw) %in% c(0,1)))) as.integer(y_raw >= 1.0) else as.integer(y_raw)

# Keep valid rows
keep <- !is.na(y)
X <- X[keep, , drop = FALSE]
y <- y[keep]

# --- Prior (majority-class) error ---
p1 <- mean(y == 1)
prior_error <- min(p1, 1 - p1)
cat(sprintf("\nLabel distribution: p(y=1)=%.4f, p(y=0)=%.4f", p1, 1-p1))
cat(sprintf("\nPrior (majority-class) error: %.4f\n", prior_error))

# --- 10-fold IDs ---
set.seed(42)
foldid <- sample(1:10, size = length(y), replace = TRUE)

# --- Fit models ---
cv_lasso <- cv.glmnet(X, y, family="binomial", alpha=1, foldid=foldid, type.measure="class")
cv_en    <- cv.glmnet(X, y, family="binomial", alpha=0.5, foldid=foldid, type.measure="class")

# --- Helper: draw prior line ---
add_prior_line <- function(prior_err) {
  abline(h = prior_err, col = "blue", lty = 2, lwd = 2)
}

out_path <- "cv_errorrate_INTERNVL_both_prior.png"
png(out_path, width = 1800, height = 950, res = 160)

# One layout: two plots on top, legend row below
layout(matrix(c(1,2,3,3), nrow = 2, byrow = TRUE), heights = c(10, 2.4))

## --- Row 1: the two plots ---
par(mar = c(5,5,6,2), mgp = c(3,1,0))

# Left panel (Lasso)
plot(cv_lasso, main = "", cex.axis = 1.1, cex.lab = 1.2)
title(main = "Lasso (alpha = 1) — Error rate vs Prior", line = 3.2, cex.main = 1.1)
add_prior_line(prior_error)

# Right panel (Elastic Net)
plot(cv_en, main = "", cex.axis = 1.1, cex.lab = 1.2)
title(main = "Elastic Net (alpha = 0.5) — Error rate vs Prior", line = 3.2, cex.main = 1.1)
add_prior_line(prior_error)

## --- Row 2: legend panel (no overlap, no clipping) ---
par(mar = c(1,0,0,0))   # compact margins for legend row
plot.new()


legend("bottomleft",
       legend = c("CV error (red)", "Dashed blue = Prior"),
       col = c("red", "blue"),
       lty = c(1, 2), lwd = c(2, 2), pch = c(19, NA),
       horiz = FALSE, bty = "n", cex = 1.05,
       inset = c(0.02, 0.22))


dev.off()
cat(sprintf("✅ Figure saved to: %s/%s\n", getwd(), out_path))


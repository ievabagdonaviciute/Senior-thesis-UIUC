# === Elastic Net / Lasso with error rate (no CSV output) ===

library(glmnet)
setwd("/home/ievab2/run_models/full_8_frames_qwen_vs_internvl/QWEN/permuted_embeddings/only_prompt")

# --- Load data ---
meta <- read.csv("prompts_meta.csv", stringsAsFactors = FALSE)
X <- as.matrix(read.csv(gzfile("embeddings_eva02_l14.csv.gz")))
y <- as.numeric(meta$correct)

keep <- !is.na(y)
X <- X[keep, , drop = FALSE]
y <- y[keep]

set.seed(42)
foldid <- sample(1:10, size = length(y), replace = TRUE)

# --- Train CV models with type.measure = "class" (=> error rate) ---
cv_lasso <- cv.glmnet(X, y, family = "binomial",
                      alpha = 1, foldid = foldid,
                      type.measure = "class")

cv_en <- cv.glmnet(X, y, family = "binomial",
                   alpha = 0.5, foldid = foldid,
                   type.measure = "class")

# --- Plot error rate vs log(lambda) ---
out_path <- "cv_errorrate_QWEN_prompt_only.png"
png(out_path, width = 1800, height = 900, res = 160)

# keep top margin, slightly smaller text
par(mfrow = c(1,2), mar = c(5,5,7,2), mgp = c(3,1,0))

# Lasso
plot(cv_lasso, main = "", cex.axis = 1.1, cex.lab = 1.2)
title(main = "Lasso (alpha = 1) — Error rate", line = 4, cex.main = 1.3)
abline(h = 0.5, lty = 2, col = "gray60", lwd = 1.1)

# Elastic Net
plot(cv_en, main = "", cex.axis = 1.1, cex.lab = 1.2)
title(main = "Elastic Net (alpha = 0.5) — Error rate", line = 4, cex.main = 1.3)
abline(h = 0.5, lty = 2, col = "gray60", lwd = 1.1)

dev.off()

# --- Print results in console ---
cat("\nLASSO (alpha=1) error@lambda.min :", round(min(cv_lasso$cvm), 4),
    "  (lambda.min=", signif(cv_lasso$lambda.min, 4), ")\n", sep = "")
cat("Elastic Net (alpha=0.5) error@lambda.min :", round(min(cv_en$cvm), 4),
    "  (lambda.min=", signif(cv_en$lambda.min, 4), ")\n", sep = "")
cat("\n✅ Plot saved to:", file.path(getwd(), out_path), "\n")

# ======================================
# 0) One-time installs (if needed)
# install.packages("glmnet")
# install.packages("data.table")
# ======================================

# 1) Load libraries + set working directory
library(glmnet)
setwd("/home/ievab2/run_models/experiment_concat_frames/embeddings/INTERNVL/image&question_embeddings")

# 2) Read metadata (y) and embeddings (X)
meta <- read.csv("image_meta.csv", stringsAsFactors = FALSE)
X <- as.matrix(read.csv(gzfile("embeddings_clip.csv.gz")))
y <- as.numeric(meta$correct)   # 1 if score==1.0, else 0

# 3) Sanity check and cleanup
keep <- !is.na(y)
X <- X[keep, , drop = FALSE]
y <- y[keep]

# 4) 10-fold cross-validation for LASSO / Elastic Net / Ridge
set.seed(42)
foldid <- sample(1:10, size = length(y), replace = TRUE)

cv1  <- cv.glmnet(X, y, family = "binomial", alpha = 1,   foldid = foldid)   # LASSO
cv05 <- cv.glmnet(X, y, family = "binomial", alpha = 0.5, foldid = foldid)   # Elastic Net
cv0  <- cv.glmnet(X, y, family = "binomial", alpha = 0,   foldid = foldid)   # Ridge

# 5) Save all plots to PNG (2x2 grid)
png("/home/ievab2/run_models/experiment_concat_frames/embeddings_concat/INTERNVL/image&question_embeddings/analysis/cv_plots.png",
    width = 1600, height = 1200, res = 180)
par(mfrow = c(2, 2))

plot(cv1);  legend("top", legend = "alpha = 1 (LASSO)",  bty = "n")
plot(cv05); legend("top", legend = "alpha = 0.5 (Elastic Net)", bty = "n")
plot(cv0);  legend("top", legend = "alpha = 0 (Ridge)",  bty = "n")

plot(log(cv1$lambda),  cv1$cvm,  pch = 19, col = "red",
     xlab = "log(Lambda)", ylab = cv1$name)
points(log(cv05$lambda), cv05$cvm, pch = 19, col = "grey")
points(log(cv0$lambda),  cv0$cvm,  pch = 19, col = "blue")
legend("topleft", legend = c("alpha=1","alpha=0.5","alpha=0"),
       pch = 19, col = c("red","grey","blue"), bty = "n")

dev.off()

# 6) Show coefficients for the best-performing model (LASSO)
cat("lambda.min (alpha=1):", cv1$lambda.min, "\n")
cat("lambda.1se (alpha=1):", cv1$lambda.1se, "\n")
print(coef(cv1, s = "lambda.min"))

# 0) One-time installs (if needed)
# install.packages("glmnet")      # only glmnet is necessary for this
# install.packages("data.table")  # faster CSV reading (optional)

# 1) Load libs + set working dir
library(glmnet)
setwd("/home/ievab2/run_models/experiment_concat_frames/embeddings_concat/INTERNVL/question_embeddings")

# 2) Read metadata (y) and embeddings (X)
meta <- read.csv("questions_meta.csv", stringsAsFactors = FALSE)
# If you have data.table installed, you can use: data.table::fread("embeddings_mpnet.csv.gz")
X <- as.matrix(read.csv(gzfile("embeddings_mpnet.csv.gz")))
y <- as.numeric(meta$correct)   # 1 if score==1.0, else 0 (as we wrote)

# 3) Basic sanity checks + clean
keep <- !is.na(y)
X <- X[keep, , drop = FALSE]
y <- y[keep]

# 4) 10-fold CV with three alphas (lasso / elastic-net / ridge)
set.seed(42)  # reproducible CV splits
foldid <- sample(1:10, size = length(y), replace = TRUE)

cv1  <- cv.glmnet(X, y, family = "binomial", alpha = 1,   foldid = foldid)   # lasso
cv05 <- cv.glmnet(X, y, family = "binomial", alpha = 0.5, foldid = foldid)   # elastic-net
cv0  <- cv.glmnet(X, y, family = "binomial", alpha = 0,   foldid = foldid)   # ridge

# 5) Save plots to PNG (2x2 grid like your example)
png("/home/ievab2/run_models/experiment_concat_frames/embeddings/INTERNVL/question_embeddings/analysis/cv_plots.png",
    width = 1600, height = 1200, res = 180)
par(mfrow = c(2, 2))

plot(cv1);  legend("top", legend = "alpha = 1",  bty = "n")
plot(cv05); legend("top", legend = "alpha = .5", bty = "n")
plot(cv0);  legend("top", legend = "alpha = 0",  bty = "n")

plot(log(cv1$lambda),  cv1$cvm,  pch = 19, col = "red",
     xlab = "log(Lambda)", ylab = cv1$name)
points(log(cv05$lambda), cv05$cvm, pch = 19, col = "grey")
points(log(cv0$lambda),  cv0$cvm,  pch = 19, col = "blue")
legend("topleft", legend = c("alpha=1","alpha=.5","alpha=0"),
       pch = 19, col = c("red","grey","blue"), bty = "n")
dev.off()

# 6) Peek at coefficients at the best lambda (text-only model)
cat("lambda.min (alpha=1):", cv1$lambda.min, "\n")
cat("lambda.1se (alpha=1):", cv1$lambda.1se, "\n")
print(coef(cv1, s = "lambda.min"))  # coefficients over embedding dimensions e0..e767

args <- commandArgs(trailingOnly=TRUE)
folder <- args[1]
.libPaths(args[3], include.site=FALSE)
if (any(normalizePath(.libPaths()) != normalizePath(args[3]))) {
  stop("CytoTRACE R libraries must all belong to the selected environment")
}
required <- c("HiClimR", "ccaPP", "nnls", "Matrix")
for (pkg in required) {
  tryCatch(loadNamespace(pkg), error=function(error) {
    stop(paste0("Cannot load R dependency ", pkg, " from ", args[3],
                "\n", conditionMessage(error),
                "\nR: ", R.version.string, " (", R.home(), ")",
                "\n.libPaths(): ", paste(.libPaths(), collapse="; ")),
         call.=FALSE)
  })
}
source(args[2])
mat <- as.matrix(Matrix::readMM(file.path(folder, "counts.mtx")))
rownames(mat) <- readLines(file.path(folder, "genes.txt"))
colnames(mat) <- readLines(file.path(folder, "cells.txt"))
stopifnot(all(is.finite(mat)), all(mat >= 0), all(mat == floor(mat)),
          all(colSums(mat) == scan(file.path(folder, "cell_sums.txt"), quiet=TRUE)),
          all(rowSums(mat) == scan(file.path(folder, "gene_sums.txt"), quiet=TRUE)))
set.seed(as.integer(args[4]))
res <- CytoTRACE(mat, batch=NULL, enableFast=FALSE, ncores=1, subsamplesize=1000)
stopifnot(identical(names(res$CytoTRACE), colnames(mat)), all(is.finite(res$CytoTRACE)))
write.csv(data.frame(cell_id=names(res$CytoTRACE), score=res$CytoTRACE),
          file.path(folder, "scores.csv"), row.names=FALSE)
capture.output(sessionInfo(), file=file.path(folder, "runtime.txt"))
write(paste(required, vapply(required, function(pkg) as.character(packageVersion(pkg)), "")),
      file=file.path(folder, "runtime.txt"), append=TRUE)

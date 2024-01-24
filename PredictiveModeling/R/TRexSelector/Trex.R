library('TRexSelector')


run_Trex <- function(df){

    res <- trex( X = as.matrix(df[,!(names(df) %in% c("y"))]), y = df$y, tFDR = 1 - 1e-20, verbose=0)

    results = res$selected_var


    return(results)
}
library(purrr)
library(irr)
library(mltools)
library(xtable)
library(dplyr)
library(ggplot2)
library(tidyr)
library(tidyverse)
library(ggcorrplot)
library(glmmTMB)


corr_summary <- function(df_kmeans, df_ocsvm, id_col = "unique_log_id", label_col = "anomaly", activity = "", print_xtable = TRUE) {
  
  # merge by ID
  merged <- merge(df_kmeans, df_ocsvm, by = id_col, suffixes = c("_kmeans", "_svm"))
  
  # Compute Cohen's Kappa
  kappa_val <- kappa2(merged[, c("anomaly_kmeans", "anomaly_svm")])$value
  
  # Compute MCC
  mcc_val <- mcc(merged$anomaly_kmeans, merged$anomaly_svm)
  
  # Get ID sets
  ids_kmeans <- merged[[id_col]][merged$anomaly_kmeans == TRUE]
  ids_svm    <- merged[[id_col]][merged$anomaly_svm == TRUE]
  
  intersection_ids <- intersect(ids_kmeans, ids_svm)
  union_ids        <- union(ids_kmeans, ids_svm)
  
  # Jaccard index
  jaccard_val <- length(intersection_ids) / length(union_ids)
  
  # summary table
  summary_df <- data.frame(
    Metric = c("Cohen Kappa", "MCC", "Intersection Count", "Union Count", "Jaccard Index"),
    Value = c(kappa_val, mcc_val, length(intersection_ids), length(union_ids), jaccard_val)
  )
  
  # print xtable if needed
  if (print_xtable) {
    print(xtable(summary_df, caption = paste("Anomaly Model Agreement Summary in ", activity), digits = 4),
          include.rownames = FALSE, caption.placement = "top")
  }
  
  return(summary_df)
}

fit_high_conf_anomaly_model <- function(
    data,
    kmeans_df,
    svm_df,
    id_col = "unique_log_id",
    anomaly_col = "anomaly",
    exclude_vars = c("anomaly", "participant_id", "unique_log_id",
                     "height_cm", "weight_kg", "time","fitness_level","endurance_level",
                     "distance_to_centroid", "cluster", "svm_score"),
    random_effect = "(time | participant_id)"
){
  
  # 1. Identify high-confidence anomaly IDs (intersection)
  k_ids <- kmeans_df[[id_col]][kmeans_df[[anomaly_col]] == TRUE]
  s_ids <- svm_df[[id_col]][svm_df[[anomaly_col]] == TRUE]
  intersection_ids <- intersect(k_ids, s_ids)
  
  # 2. Create high anomaly flag
  data$highAnomaly <- ifelse(data[[id_col]] %in% intersection_ids, 1, 0)
  
  print(paste0("Num of Anomalies: ", nrow(data[data$highAnomaly == 1, ])))
  print(paste0("Num of non-anomalies: ", nrow(data[data$highAnomaly == 0, ])))
  
  # 3. Determine fixed effects
  fixed_effects <- setdiff(names(data), exclude_vars)
  
  # 4. Build formula
  formula_str <- paste0(
    "highAnomaly ~ ",
    paste(fixed_effects, collapse = " + "),
    " + ",
    random_effect
  )
  model_formula <- as.formula(formula_str)
  
  # 5. Fit mixed logistic regression
  model <- glmmTMB(
    model_formula,
    data = data,
    family = binomial(link = "logit")
  )
  
  # 6. Output list
  list(
    formula = model_formula,
    model = model,
    summary = summary(model),
    high_conf_ids = intersection_ids
  )
}


model_to_xtable <- function(
    model,
    digits = 3,
    sci_threshold = 0.001,
    caption_fixed = "Fixed effects: High-confidence anomaly mixed-effects logistic regression",
    caption_random = "Random effects (variance components)",
    label_fixed = NULL,
    label_random = NULL
){
  s <- summary(model)
  
  #-------------------------
  # Fixed effects
  #-------------------------
  coefs <- s$coefficients$cond
  
  OR <- exp(coefs[, 1])
  lower_CI <- exp(coefs[, 1] - 1.96 * coefs[, 2])
  upper_CI <- exp(coefs[, 1] + 1.96 * coefs[, 2])
  
  fixed_table <- data.frame(
    Estimate = coefs[, 1],
    StdErr = coefs[, 2],
    z_value = coefs[, 3],
    p_value = coefs[, 4],
    OR = OR,
    CI_lower = lower_CI,
    CI_upper = upper_CI,
    stringsAsFactors = FALSE
  )
  
  # Apply rounding
  fixed_table[, c("StdErr", "z_value", "OR", "CI_lower", "CI_upper")] <-
    round(fixed_table[, c("StdErr", "z_value", "OR", "CI_lower", "CI_upper")], digits)
  
  # Convert to scientific notation
  to_sci <- function(x) {
    ifelse(abs(x) < sci_threshold,
           formatC(x, format = "e", digits = 2),
           formatC(x, format = "f", digits = digits))
  }
  
  fixed_table$Estimate <- to_sci(fixed_table$Estimate)
  fixed_table$p_value <- to_sci(fixed_table$p_value)
  
  xt_fixed <- xtable::xtable(
    fixed_table,
    caption = caption_fixed,
    label = label_fixed,
    align = c("l", rep("r", ncol(fixed_table)))
  )
  
  #-------------------------
  # Random Effect
  #-------------------------
  vc <- VarCorr(model)
  
  
  #-------------------------
  # Return list
  #-------------------------
  return(list(
    fixed = xt_fixed
  ))
}

pair_corr_summary = function(datas, suffix, id_col = "unique_log_id", label_col = "anomaly", activity = ""){
  ## ---- input validation ----
  if (!is.list(datas)) stop("datas must be a list.")
  if (!is.vector(suffix)) stop("suffix must be a vector.")
  if (length(datas) != length(suffix)) stop("datas and suffix must have same length.")
  
  # ---- Rename each dataframe ----
  renamed_list <- map2(datas, suffix, function(df, suf) {
    df %>% rename(
      !!paste0("anomaly_", suf) := all_of(label_col)
    )
  })
  
  # ---- Merge all data ----
  anomaly.results <- reduce(renamed_list, full_join, by = id_col)
  print(head(anomaly.results))
  
  # ---- Prepare pairwise combinations ----
  anomaly_cols <- grep("^anomaly_", names(anomaly.results), value = TRUE)
  print(anomaly_cols)
  combs <- combn(anomaly_cols, 2, simplify = FALSE)
  
  # ---- Compute pairwise metrics ----
  results <- map_df(combs, function(x) {
    col_a <- x[1]
    col_b <- x[2]
    
    m1 <- anomaly.results[[col_a]]
    m2 <- anomaly.results[[col_b]]
    
    # Cohen's Kappa (requires 2-column factor data.frame)
    df_pair <- data.frame(
      a = factor(m1, levels = c(FALSE, TRUE)),
      b = factor(m2, levels = c(FALSE, TRUE))
    )
    kap <- irr::kappa2(df_pair)$value
    
    # MCC
    mcc_val <- mcc(m1, m2)
    
    # Jaccard
    ids1 <- anomaly.results[[id_col]][m1 == TRUE]
    ids2 <- anomaly.results[[id_col]][m2 == TRUE]
    jaccard <- length(intersect(ids1, ids2)) / length(union(ids1, ids2))
    
    tibble(
      model_a = col_a,
      model_b = col_b,
      kappa = kap,
      mcc = mcc_val,
      jaccard = jaccard
    )
  })
  
  return(list(
    anomalies = anomaly.results,
    summary = results
  ))
}

summary_to_corr_matrix <- function(data, col_name, plot_name, activity){
  ## check data types 
  if(!is.data.frame(data) || !is.character(col_name) || !is.character(plot_name) || !is.character(activity)){
    stop("Invalid parameter for data and col_name! ")
  }else if(!(col_name %in% colnames(data))){
    ### case when column name not exist
    stop(paste0("Column name is not exist", col_name))
  }
  
  heat_df <- data %>%
    select(model_a, model_b, all_of(col_name)) %>%
    mutate(
      value = as.numeric(.data[[col_name]]),
      # 對角線直接設 1
      value = ifelse(model_a == model_b, 1, value)
    )
  
  
  p <- ggplot(heat_df, aes(x = model_b, y = model_a, fill = value)) +
    geom_tile(color = "white") +
    geom_text(aes(label = sprintf("%.2f", value)), size = 3) +
    scale_fill_gradient(low = "white", high = "red", limits = c(0,1)) +
    labs(
      title = paste0("Heatmap of ", col_name),
      fill = col_name
    ) +
    theme_minimal(base_size = 12) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      plot.margin = margin(20,20,20,20)
    )
  
  print(p)
  
  ggsave(paste0(activity, "_", plot_name,".png"), plot = p, width = 7, height = 6, dpi = 300)
}

high.confidence.anomaly.detection = function(
    data, 
    data_list, 
    suffix, 
    id_col = "unique_log_id", 
    label_col = "anomaly",
    exclude_vars = c("anomaly", "participant_id", "unique_log_id",
                     "height_cm", "weight_kg", "time", "fitness_level", "endurance_level",
                     "distance_to_centroid", "cluster", "svm_score"),
    random_effect = "(time | participant_id)"
){
  
  # -- Check inputs are valid --
  if (!is.list(data_list)) stop("data_list must be a list.")
  if (!is.vector(suffix)) stop("suffix must be a vector.")
  if (length(data_list) != length(suffix)) stop("data_list and suffix must match in length.")
  
  # -- Rename label_col to anomaly_suffix in each dataframe --
  renamed_list <- map2(data_list, suffix, function(df, suf) {
    df %>% rename(
      !!paste0("anomaly_", suf) := all_of(label_col)
    )
  })
  
  # -- Merge all anomaly dataframes by id_col --
  anomaly.results <- reduce(renamed_list, full_join, by = id_col)
  
  # -- Keep only anomaly_* columns --
  anomaly_cols <- paste0("anomaly_", suffix)
  print(anomaly_cols)
  anomaly.results <- anomaly.results[, c(id_col, anomaly_cols)]
  
  print(head(anomaly.results, 10))
  
  # -- Compute high confidence anomaly:
  #    If sum of anomaly_* columns > 1, then highAnomaly = 1 else 0
  anomaly.results$highAnomaly <- as.integer(
    rowSums(anomaly.results[, anomaly_cols], na.rm = TRUE) > 1
  )
  print(paste0("High confidence count: ", nrow(anomaly.results[anomaly.results$highAnomaly == 1, ])))
  print(paste0("Non High confidence count: ", nrow(anomaly.results[anomaly.results$highAnomaly == 0, ])))
  
  # -- Keep only ID and highAnomaly for merging --
  lme.df <- anomaly.results[, c(id_col, "highAnomaly")]
  
  # -- Merge with original data --
  complete.df <- merge(data, lme.df, by = id_col)
  
  # -- Determine fixed effects (exclude unwanted variables) --
  fixed_effects <- setdiff(names(data), exclude_vars)
  
  # -- Build mixed logistic regression formula --
  formula_str <- paste0(
    "highAnomaly ~ ",
    paste(fixed_effects, collapse = " + "),
    " + ",
    random_effect
  )
  
  model_formula <- as.formula(formula_str)
  
  # 5. Fit mixed logistic regression
  model <- glmmTMB(
    model_formula,
    data = complete.df,
    family = binomial(link = "logit")
  )
  
  return(list(
    model = model,
    summary = summary(model),
    formula = as.formula(formula_str)
  ))
}

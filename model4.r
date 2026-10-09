library(brms)
library(tidyverse)
library(tidybayes)
library(posterior)
library(bayesplot)
library(patchwork)
library(readxl)

df <- read_excel("experimental_data.xlsx") %>%
  mutate(
    cohort         = as.factor(experiment),
    # Authentic as reference
    utterance_type = factor(utterance_type,
                            levels = c("Authentic", "Partially Synthetic", "Synthetic")),
    # Effect coding (+0.5 / -0.5); positive = direction of increased suspicion
    provenance  = if_else(condition_provenance == "FALSE",                0.5, -0.5),
    valence     = if_else(condition_valence    == "negative",            0.5, -0.5),
    instruction = if_else(condition_instruction == "monitor_attacks",    0.5, -0.5),
    # Participant vote: ordered factor
    vote_ord    = ordered(vote_ord, levels = c(0L, 1L, 2L),
                          labels = c("Authentic", "Partial", "Synthetic")),
    vote_correct_01 = as.integer(vote_correct == 100L),
    participant_id  = as.factor(participant_id),
    utterance_filename = as.factor(str_replace(utterance_filename, "partial-cf", "partial_cf")),
    # Confidence as ordered factor for its own outcome model
    confidence_ord  = ordered(rating_confidence_pct,
                              levels = c(0, 25, 50, 75, 100))
  )

z <- function(x) (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)

df <- df %>%
  mutate(
    quality_z    = z(quality_score),
    confidence_z = z(rating_confidence_pct),
    # Dimension ratings on original -2:2 scale as numeric, so we can have correlations
    across(
      c(rating_mechanicalness_rev, rating_expressiveness,
        rating_intelligibility, rating_clarity, rating_calmness),
      as.numeric,
      .names = "{.col}_num"
    )
  )

NUTS       <- list(chains = 4, iter = 8000, warmup = 2000, seed = 42,
                   backend = "cmdstanr", cores = 4, file_refit = "on_change",
                   control = list(adapt_delta = 0.95))
NUTS_tight <- modifyList(NUTS, list(control = list(adapt_delta = 0.99)))

# Priors
priors_quality <- c(
  prior(normal(0, 1), class = b),
  prior(normal(0, 1), class = sigma),
  prior(normal(0, 1), class = sd)
)

priors_vote <- c(
  prior(normal(0, 3), class = Intercept),
  prior(normal(0, 1), class = b),
  prior(normal(0, 1), class = sd)
)

# Confidence outcome: cumulative logit, five categories.
priors_confidence <- c(
  prior(normal(0, 3), class = Intercept),
  prior(normal(0, 1), class = b),
  prior(normal(0, 1), class = sd)
)

priors_correct <- c(
  prior(normal(0, 1.5), class = Intercept),
  prior(normal(0, 1),   class = b),
  prior(normal(0, 1),   class = sd)
)

#  Formulas 
# A-path: does utterance type (vs Authentic) predict perceived quality,
# and do trust cues moderate this effect?
shared_predictors <- paste(
  "utterance_type * provenance +",
  "utterance_type * valence + utterance_type * instruction +",
  "cohort + (1 + utterance_type + provenance | cohort:participant_id) + (1 + provenance | utterance_filename)"
)

quality_formula    <- as.formula(paste("quality_z ~",    shared_predictors))
# B-path + direct effect: does quality predict vote category after controlling
# for utterance type directly?
vote_formula       <- as.formula(paste(
  "vote_ord ~ utterance_type * provenance +",
  "utterance_type * valence + utterance_type * instruction +",
  "quality_z +",
  "cohort + (1 + utterance_type + provenance | cohort:participant_id) + (1 + provenance | utterance_filename)"
))

confidence_formula <- as.formula(paste("confidence_ord ~", shared_predictors,
                                       "+ quality_z"))

# Calibration model: does confidence predict correctness over and above vote
# category, and does that relationship differ by vote category?
correct_formula <- as.formula(
  "vote_correct_01 ~ vote_ord * confidence_z + quality_z +
   cohort + (1 | cohort:participant_id) + (1 | utterance_filename)"
)

# Prior predictive checks
m_quality_prior <- do.call(brm, c(list(
  formula      = quality_formula,
  data         = df,
  family       = gaussian(),
  prior        = priors_quality,
  sample_prior = "only",
  file         = "m_quality_prior"
), NUTS_tight))

m_vote_prior <- do.call(brm, c(list(
  formula      = vote_formula,
  data         = df,
  family       = cumulative("logit", threshold = "flexible"),
  prior        = priors_vote,
  sample_prior = "only",
  file         = "m_vote_prior"
), NUTS_tight))

m_confidence_prior <- do.call(brm, c(list(
  formula      = confidence_formula,
  data         = df,
  family       = cumulative("logit", threshold = "flexible"),
  prior        = priors_confidence,
  sample_prior = "only",
  file         = "m_confidence_prior"
), NUTS_tight))

m_correct_prior <- do.call(brm, c(list(
  formula      = correct_formula,
  data         = df,
  family       = bernoulli("logit"),
  prior        = priors_correct,
  sample_prior = "only",
  file         = "m_correct_prior"
), NUTS))

# Fitted models
# A-path: utterance type -> perceived quality
m_quality <- do.call(brm, c(list(
  formula = quality_formula,
  data    = df,
  family  = gaussian(),
  prior   = priors_quality,
  file    = "m_quality"
), NUTS_tight))

# B-path + direct effect: vote as a function of quality and utterance type
m_vote <- do.call(brm, c(list(
  formula = vote_formula,
  data    = df,
  family  = cumulative("logit", threshold = "flexible"),
  prior   = priors_vote,
  file    = "m_vote"
), NUTS_tight))

m_confidence <- do.call(brm, c(list(
  formula = confidence_formula,
  data    = df,
  family  = cumulative("logit", threshold = "flexible"),
  prior   = priors_confidence,
  file    = "m_confidence"
), NUTS_tight))

m_correct <- do.call(brm, c(list(
  formula = correct_formula,
  data    = df,
  family  = bernoulli("logit"),
  prior   = priors_correct,
  file    = "m_correct"
), NUTS))

# Convergence diagnostics
check_convergence <- function(model, name) {
  d        <- as_draws_array(model) %>%
                subset_draws(variable = "^(b_|sd_|cor_|sigma|rescor|nu)", regex = TRUE) %>%
                posterior::summarise_draws("rhat", "ess_bulk", "ess_tail")
  rhat     <- max(d$rhat,     na.rm = TRUE)
  bulk_ess <- min(d$ess_bulk, na.rm = TRUE)
  tail_ess <- min(d$ess_tail, na.rm = TRUE)
  divs     <- nuts_params(model) %>% filter(Parameter == "divergent__") %>%
                pull(Value) %>% sum()
  flag <- if_else(rhat > 1.01 | divs > 0 | bulk_ess < 400 | tail_ess < 400, " ⚠", "")
  message(sprintf("%-20s | Rhat=%.3f | divs=%d | bulk_ESS=%.0f | tail_ESS=%.0f%s",
                  name, rhat, divs, bulk_ess, tail_ess, flag))
}

main_models      <- list(m_quality, m_vote, m_confidence, m_correct)
main_model_names <- c("quality", "vote", "confidence", "correct")
walk2(main_models, main_model_names, check_convergence)

# Trace plots
walk2(main_models, main_model_names, \(mod, nm)
  ggsave(paste0("trace_", nm, ".png"),
         mcmc_trace(as_draws_array(mod), regex_pars = "^b_") +
           theme_minimal(base_size = 8),
         width = 12, height = 6, dpi = 120))

# Posterior predictive checks
wrap_plots(list(
  pp_check(m_quality,    ndraws = 100) + ggtitle("Quality") + theme_minimal(),
  pp_check(m_vote,       type = "bars_grouped", group = "cohort", ndraws = 100) +
    ggtitle("Vote by cohort") + theme_minimal(),
  pp_check(m_confidence, type = "bars_grouped", group = "cohort", ndraws = 100) +
    ggtitle("Confidence by cohort") + theme_minimal(),
  pp_check(m_correct,    type = "bars_grouped", group = "cohort", ndraws = 100) +
    ggtitle("Correctness by cohort") + theme_minimal()
), ncol = 2)
ggsave("pp_checks_main.png", width = 15, height = 10, dpi = 150)

# Draw extraction and indirect effects
draws_quality    <- as_draws_df(m_quality)
draws_vote       <- as_draws_df(m_vote)
draws_confidence <- as_draws_df(m_confidence)
draws_correct    <- as_draws_df(m_correct)

# A-paths: effect of utterance type on quality relative to Authentic baseline
apath_quality_partial   <- draws_quality[["b_utterance_typePartiallySynthetic"]]
apath_quality_synthetic <- draws_quality[["b_utterance_typeSynthetic"]]

# B-path: effect of perceived quality on vote
bpath_quality_to_vote <- draws_vote$b_quality_z

#  Indirect effects (a * b) 
indirect_partial   <- apath_quality_partial   * bpath_quality_to_vote
indirect_synthetic <- apath_quality_synthetic * bpath_quality_to_vote

summarise_effects <- function(...) {
  vecs <- list(...)
  tibble(
    mean  = map_dbl(vecs, mean),
    lo95  = map_dbl(vecs, quantile, 0.025),
    hi95  = map_dbl(vecs, quantile, 0.975),
    p_pos = map_dbl(vecs, \(x) mean(x > 0))
  )
}

indirect_effects <- bind_cols(
  tibble(utterance_type = c("Partially Synthetic", "Synthetic")),
  summarise_effects(indirect_partial, indirect_synthetic)
)
cat("\nIndirect effects of utterance type on vote via quality\n")
print(indirect_effects)

#  A-paths: utterance type -> confidence 
apath_conf_partial   <- draws_confidence[["b_utterance_typePartiallySynthetic"]]
apath_conf_synthetic <- draws_confidence[["b_utterance_typeSynthetic"]]
bpath_quality_to_conf <- draws_confidence$b_quality_z

cat("\nEffect of utterance type on confidence (log-odds of higher rating)\n")
bind_cols(
  tibble(utterance_type = c("Partially Synthetic", "Synthetic")),
  summarise_effects(apath_conf_partial, apath_conf_synthetic)
) %>% print()

cat("\nEffect of quality on confidence (B-path, log-odds)\n")
summarise_effects(bpath_quality_to_conf) %>% print()

# Residual correlation between vote and confidence
participant_intercepts <- function(draws) {
  m <- as_draws_matrix(draws)
  m[, grepl("^r_cohort:participant_id\\[.*,Intercept\\]$", colnames(m)), drop = FALSE]
}
ranef_vote <- participant_intercepts(draws_vote)
ranef_conf <- participant_intercepts(draws_confidence)
shared_ids <- intersect(colnames(ranef_vote), colnames(ranef_conf))

residual_cor_draws <- map_dbl(seq_len(nrow(ranef_vote)), \(i)
  cor(as.numeric(ranef_vote[i, shared_ids]), as.numeric(ranef_conf[i, shared_ids]),
      method = "spearman"))

cat(sprintf("\nResidual correlation (Spearman) between vote and confidence random intercepts:\n"))
cat(sprintf("r = %.3f [%.3f, %.3f]\n", mean(residual_cor_draws),
            quantile(residual_cor_draws, 0.025), quantile(residual_cor_draws, 0.975)))

#  Calibration: does confidence predict correctness beyond vote?
correctness_effects <- bind_cols(
  tibble(parameter = c(
    "Linear vote trend",
    "Quadratic vote trend",
    "Confidence (main effect)",
    "Quality (main effect)",
    "Linear vote x Confidence",
    "Quadratic vote x Confidence"
  )),
  summarise_effects(
    draws_correct$`b_vote_ord.L`,
    draws_correct$`b_vote_ord.Q`,
    draws_correct$b_confidence_z,
    draws_correct$b_quality_z,
    draws_correct$`b_vote_ord.L:confidence_z`,
    draws_correct$`b_vote_ord.Q:confidence_z`
  )
)
cat("\nCalibration model: vote + confidence -> correctness\n")
print(correctness_effects)

# Quality dimensions model
# Five perceptual dimensions modelled jointly as student-t to:
# (a) allow correlated residuals between dimensions
# (b) handle heavier tails than Gaussian
dim_vars   <- c("rating_mechanicalness_rev_num", "rating_expressiveness_num",
                "rating_intelligibility_num",    "rating_clarity_num",
                "rating_calmness_num")
dim_resp   <- c("mechanicalness", "expressiveness", "intelligibility",
                "clarity", "calmness")
dim_labels <- c("Mechanicalness (reversed)", "Expressiveness",
                "Intelligibility", "Clarity", "Calmness")

dim_formula <- function(outcome) as.formula(paste(
  outcome, "~ utterance_type * provenance +",
  "utterance_type * valence + utterance_type * instruction +",
  "cohort + (1 + utterance_type + provenance | cohort:participant_id) + (1 + provenance | utterance_filename)"
))

priors_dims <- do.call(c, lapply(dim_resp, \(r) c(
  prior_string("normal(0, 1)", class = "b",     resp = r),
  prior_string("normal(0, 1)", class = "sigma", resp = r),
  prior_string("normal(0, 1)", class = "sd",    resp = r)
)))

df_dims <- df %>% rename_with(~ dim_resp, all_of(dim_vars))

m_dims <- brm(
  formula = do.call(mvbf, c(map(dim_resp, dim_formula), list(rescor = TRUE))),
  data    = df_dims,
  family  = student(),
  prior   = priors_dims,
  file    = "m_dims",
  chains  = 4, iter = 8000, warmup = 2000, seed = 42,
  backend = "cmdstanr", cores = 4, file_refit = "on_change",
  control = list(adapt_delta = 0.99)
)

check_convergence(m_dims, "dims")

ggsave("trace_dims.png",
       mcmc_trace(as_draws_array(m_dims), regex_pars = "^b_") +
         theme_minimal(base_size = 8),
       width = 12, height = 6, dpi = 120)

pp_dim_panels <- map(dim_resp, \(r)
  pp_check(m_dims, resp = r, ndraws = 100) +
    ggtitle(dim_labels[match(r, dim_resp)]) + theme_minimal())
wrap_plots(pp_dim_panels, ncol = 3)
ggsave("pp_checks_dims.png", width = 15, height = 8, dpi = 150)

# Effect of utterance type on each dimension (vs Authentic)
draws_dims <- as_draws_df(m_dims)

dimension_effects <- map2_dfr(dim_resp, dim_labels, \(resp, label) {
  partial_col_dim   <- paste0("b_", resp, "_utterance_typePartiallySynthetic")
  synthetic_col_dim <- paste0("b_", resp, "_utterance_typeSynthetic")
  bind_rows(
    summarise_effects(draws_dims[[partial_col_dim]])   %>%
      mutate(dimension = label, contrast = "Partially Synthetic vs Authentic"),
    summarise_effects(draws_dims[[synthetic_col_dim]]) %>%
      mutate(dimension = label, contrast = "Synthetic vs Authentic")
  )
}) %>% select(dimension, contrast, everything())

cat("\nEffect of utterance type on quality dimensions\n")
print(dimension_effects)

cat("\nResidual correlations between quality dimensions\n")
print(VarCorr(m_dims)$residual__$cor)

L <- contr.poly(3)
conf_slopes <- map(1:3, \(k) draws_correct$b_confidence_z +
  L[k, 1] * draws_correct$`b_vote_ord.L:confidence_z` +
  L[k, 2] * draws_correct$`b_vote_ord.Q:confidence_z`)

# Vote thresholds
tau1 <- draws_vote[["b_Intercept[1]"]]
tau2 <- draws_vote[["b_Intercept[2]"]]
cat("\nVote thresholds (logit scale)\n")
bind_cols(
  tibble(parameter = c("tau1 (Authentic | Partial)", "tau2 (Partial | Synthetic)", "tau2 - tau1")),
  summarise_effects(tau1, tau2, tau2 - tau1)
) %>% print()

# Trust cues
moderators <- c("provenance", "valence", "instruction")

moderator_coefs <- \(draws) {
  cols <- grep("^b_.*(provenance|valence|instruction)", names(draws), value = TRUE)
  bind_cols(tibble(parameter = cols),
            do.call(summarise_effects, map(cols, \(col) draws[[col]])))
}
cat("\nTrust cue coefficients, quality model (SD units)\n")
print(moderator_coefs(draws_quality))
cat("\nTrust cue coefficients, vote model (log-odds)\n")
print(moderator_coefs(draws_vote))

m_total <- do.call(brm, c(list(
  formula = update(vote_formula, . ~ . - quality_z),
  data    = df,
  family  = cumulative("logit", threshold = "flexible"),
  prior   = priors_vote,
  file    = "m_total"
), NUTS_tight))


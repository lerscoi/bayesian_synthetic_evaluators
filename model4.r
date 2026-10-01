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
    # Authentic as reference, no equal-spacing assumption
    utterance_type = factor(utterance_type,
                            levels = c("Authentic", "Partially Synthetic", "Synthetic")),
    # Effect coding (+0.5 / -0.5); positive = direction of increased suspicion
    provenance  = if_else(condition_provenance,                          0.5, -0.5),
    valence     = if_else(condition_valence    == "negative",            0.5, -0.5),
    instruction = if_else(condition_instruction == "monitor_attacks",    0.5, -0.5),
    # Participant vote: ordered factor for cumulative logit outcome model
    vote_ord    = ordered(vote_ord, levels = c(0L, 1L, 2L),
                              labels = c("Authentic", "Partial", "Synthetic")),
    vote_correct_01 = as.integer(vote_correct == 100L),
    participant_id  = as.factor(participant_id)
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
                   backend = "cmdstanr", cores = 4,
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

priors_correct <- c(
  prior(normal(0, 1.5), class = Intercept),
  prior(normal(0, 1),   class = b),
  prior(normal(0, 1),   class = sd)
)

# Formulas
# A-path: does utterance type (vs Authentic) predict perceived quality,
# and do trust cues moderate this effect?
quality_formula <- paste(
  "quality_z ~ utterance_type * provenance +",
  "utterance_type * valence + utterance_type * instruction +",
  "(1 | cohort / participant_id)"
)

# B-path + direct effect: does quality predict vote category after controlling
# for utterance type directly and confidence as a covariate?
vote_formula <- paste(
  "vote_ord ~ utterance_type * provenance +",
  "utterance_type * valence + utterance_type * instruction +",
  "quality_z + confidence_z +",
  "(1 | cohort / participant_id)"
)

# Prior predictive checks
m_quality_prior <- do.call(brm, c(list(
  formula      = as.formula(quality_formula),
  data         = df,
  family       = gaussian(),
  prior        = priors_quality,
  sample_prior = "only",
  file         = "m_quality_prior"
), NUTS_tight))

m_vote_prior <- do.call(brm, c(list(
  formula      = as.formula(vote_formula),
  data         = df,
  family       = cumulative("logit", threshold = "flexible"),
  prior        = priors_vote,
  sample_prior = "only",
  file         = "m_vote_prior"
), NUTS_tight))

m_correct_prior <- do.call(brm, c(list(
  formula      = vote_correct_01 ~ vote_ord + (1 | cohort / participant_id),
  data         = df,
  family       = bernoulli("logit"),
  prior        = priors_correct,
  sample_prior = "only",
  file         = "m_correct_prior"
), NUTS))

# Fitted models
# A-path: utterance type -> perceived quality
m_quality <- do.call(brm, c(list(
  formula = as.formula(quality_formula),
  data    = df,
  family  = gaussian(),
  prior   = priors_quality,
  file    = "m_quality"
), NUTS_tight))

# B-path + direct effect: vote as a function of quality and utterance type
m_vote <- do.call(brm, c(list(
  formula = as.formula(vote_formula),
  data    = df,
  family  = cumulative("logit", threshold = "flexible"),
  prior   = priors_vote,
  file    = "m_vote"
), NUTS_tight))

m_correct <- do.call(brm, c(list(
  formula = vote_correct_01 ~ vote_ord + (1 | cohort / participant_id),
  data    = df,
  family  = bernoulli("logit"),
  prior   = priors_correct,
  file    = "m_correct"
), NUTS))

# Convergence diagnostics
check_convergence <- function(model, name) {
  s        <- summary(model)
  fixed    <- if (inherits(model, "mvbrmsfit")) do.call(rbind, s$fixed) else s$fixed
  rhat     <- max(fixed[, "Rhat"],     na.rm = TRUE)
  bulk_ess <- min(fixed[, "Bulk_ESS"], na.rm = TRUE)
  tail_ess <- min(fixed[, "Tail_ESS"], na.rm = TRUE)
  divs     <- nuts_params(model) %>% filter(Parameter == "divergent__") %>%
                pull(Value) %>% sum()
  flag <- if_else(rhat > 1.01 | divs > 0 | bulk_ess < 400 | tail_ess < 400, " ⚠", "")
  message(sprintf("%-20s | Rhat=%.3f | divs=%d | bulk_ESS=%.0f | tail_ESS=%.0f%s",
                  name, rhat, divs, bulk_ess, tail_ess, flag))
}

main_models      <- list(m_quality, m_vote, m_correct)
main_model_names <- c("quality", "vote", "correct")
walk2(main_models, main_model_names, check_convergence)

# Trace plots
walk2(main_models, main_model_names, \(mod, nm)
  ggsave(paste0("trace_", nm, ".png"),
         mcmc_trace(as_draws_array(mod), regex_pars = "^b_") +
           theme_minimal(base_size = 8),
         width = 12, height = 6, dpi = 120))

# Posterior predictive checks
wrap_plots(list(
  pp_check(m_quality, ndraws = 100) + ggtitle("Quality mediator") + theme_minimal(),
  pp_check(m_vote, type = "bars_grouped", group = "cohort", ndraws = 100) +
    ggtitle("Vote outcome by cohort") + theme_minimal()
), ncol = 2)
ggsave("pp_checks_main.png", width = 15, height = 6, dpi = 150)

# Draw extraction and indirect effects
draws_quality <- as_draws_df(m_quality)
draws_vote    <- as_draws_df(m_vote)
draws_correct <- as_draws_df(m_correct)

# A-paths: effect of utterance type on quality relative to Authentic baseline
partial_col   <- grep("utterance_type.*Partial", names(draws_quality), value = TRUE)[1]
synthetic_col <- grep("utterance_type.*Synthetic", names(draws_quality), value = TRUE)[1]
message("A-path columns: ", partial_col, " | ", synthetic_col)

apath_quality_partial   <- draws_quality[[partial_col]]
apath_quality_synthetic <- draws_quality[[synthetic_col]]

# B-path: effect of perceived quality on vote
bpath_quality_to_vote <- draws_vote$b_quality_z

# Indirect effects = a * b
indirect_partial   <- apath_quality_partial   * bpath_quality_to_vote
indirect_synthetic <- apath_quality_synthetic * bpath_quality_to_vote

summarise_draws <- function(...) {
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
  summarise_draws(indirect_partial, indirect_synthetic)
)

cat("\n Indirect effects of utterance type on vote via quality \n")
cat("(log-odds scale; positive = shift toward Synthetic vote)\n\n")
print(indirect_effects)

correctness_by_vote <- bind_cols(
  tibble(contrast = c(
    "Linear: correctness trend Auth < Partial < Synth",
    "Quadratic: Partial vote correctness vs Auth/Synth average"
  )),
  summarise_draws(draws_correct$`b_vote_ord.L`, draws_correct$`b_vote_ord.Q`)
)

cat("\nVote category -> correctness (log-odds) \n\n")
print(correctness_by_vote)

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
  "(1 | cohort / participant_id)"
))

priors_dims <- do.call(c, lapply(dim_resp, \(r) c(
  prior_string("normal(0, 1)", class = "b",     resp = r),
  prior_string("normal(0, 1)", class = "sigma", resp = r),
  prior_string("normal(0, 1)", class = "sd",    resp = r)
)))

df_dims <- df %>%
  rename_with(~ dim_resp, all_of(dim_vars))

m_dims <- brm(
  formula = do.call(mvbf, c(
    map(dim_resp, dim_formula),
    list(rescor = TRUE)
  )),
  data    = df_dims,
  family  = student(),
  prior   = priors_dims,
  file    = "m_dims",
  chains  = 4, iter = 8000, warmup = 2000, seed = 42,
  backend = "cmdstanr", cores = 4,
  control = list(adapt_delta = 0.99)
)

check_convergence(m_dims, "dims")

ggsave("trace_dims.png",
       mcmc_trace(as_draws_array(m_dims), regex_pars = "^b_") +
         theme_minimal(base_size = 8),
       width = 12, height = 6, dpi = 120)

# PP checks per dimension
pp_dim_panels <- map(dim_resp, \(r)
  pp_check(m_dims, resp = r, ndraws = 100) +
    ggtitle(dim_labels[match(r, dim_resp)]) + theme_minimal())
wrap_plots(pp_dim_panels, ncol = 3)
ggsave("pp_checks_dims.png", width = 15, height = 8, dpi = 150)

# Effect of utterance type on each dimension (vs Authentic)
draws_dims <- as_draws_df(m_dims)

dimension_effects <- map2_dfr(dim_resp, dim_labels, \(resp, label) {
    partial_col_dim   <- grep(paste0("b_", resp, ".*Partial"),   names(draws_dims), value = TRUE)[1]
    synthetic_col_dim <- grep(paste0("b_", resp, ".*Synthetic"), names(draws_dims), value = TRUE)[1]
    partial_draws     <- draws_dims[[partial_col_dim]]
    synthetic_draws   <- draws_dims[[synthetic_col_dim]]
  bind_rows(
    summarise_draws(partial_draws)   %>% mutate(dimension = label, contrast = "Partially Synthetic vs Authentic"),
    summarise_draws(synthetic_draws) %>% mutate(dimension = label, contrast = "Synthetic vs Authentic")
  )
}) %>% select(dimension, contrast, everything())

cat("\nEffect of utterance type on quality dimensions \n\n")
print(dimension_effects)

cat("\nResidual correlations between quality dimensions \n")
print(VarCorr(m_dims)$residual__$cor)
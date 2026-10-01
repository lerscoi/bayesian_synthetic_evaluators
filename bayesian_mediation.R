install.packages("brms")

library(brms)
library(tidyverse)
library(tidybayes)
library(posterior)
library(bayesplot)
library(patchwork)
library(readxl)

df <- read_excel("experimental_data.xlsx") %>%
  mutate(
    vote_ord = ordered(vote_ord, levels = c(0, 1, 2),
                       labels = c("Authentic", "Partial", "Synthetic")),
    participant_id = as.factor(participant_id)
  )

message("N trials: ", nrow(df), "; N participants: ", n_distinct(df$participant_id))
message("Vote distribution: ", paste(table(df$vote_ord), collapse = " / "))

NUTS <- list(chains = 4, iter = 4000, warmup = 2000, seed = 42, backend = "cmdstanr")

priors_gaussian <- c(
  prior(normal(0, 1), class = b),
  prior(normal(0, 1), class = sd),
  prior(normal(0, 1), class = sigma)
)

priors_vote <- c(
  prior(normal(0, 1), class = b),
  prior(normal(0, 1.5), class = Intercept),
  prior(normal(0, 1), class = sd)
)

priors_correct <- c(
  prior(normal(0, 1), class = b),
  prior(normal(0, 1), class = sd)
)

m_quality <- do.call(brm, c(list(
  formula = quality_score_composite_z ~ utype_num_z + provenance_z + (1 | participant_id),
  data = df, family = gaussian(), prior = priors_gaussian, file = "m_quality"), NUTS))

m_revisions <- do.call(brm, c(list(
  formula = revision_count ~ utype_num_z + provenance_z + (1 | participant_id),
  data = df, family = gaussian(), prior = priors_gaussian, file = "m_revisions"), NUTS))

m_duration <- do.call(brm, c(list(
  formula = log_duration_z ~ utype_num_z + provenance_z + (1 | participant_id),
  data = df, family = gaussian(), prior = priors_gaussian, file = "m_duration"), NUTS))

m_confidence <- do.call(brm, c(list(
  formula = rating_confidence_z ~ utype_num_z + provenance_z + (1 | participant_id),
  data = df, family = gaussian(), prior = priors_gaussian, file = "m_confidence"), NUTS))

m_vote <- do.call(brm, c(list(
  formula = vote_ord ~ utype_num_z +
    quality_score_composite_z +
    revision_count + log_duration_z + rating_confidence_z +
    dprime_z + criterion_z +
    provenance_z + (1 | participant_id),
  data = df, family = cumulative("logit"),
  prior = priors_vote, file = "m_vote"), NUTS))

m_correct <- do.call(brm, c(list(
  formula = vote_correct_01 ~ utype_num_z +
    quality_score_composite_z +
    revision_count + log_duration_z + rating_confidence_z +
    dprime_z + criterion_z +
    provenance_z + (1 | participant_id),
  data = df, family = bernoulli("logit"),
  prior = priors_correct, file = "m_correct"), NUTS))

dim_vars  <- c("rating_mechanicalness_rev_z", "rating_expressiveness_z",
               "rating_intelligibility_z", "rating_clarity_z", "rating_calmness_z")
dim_files  <- c("m_dim_mechanical", "m_dim_expressive", "m_dim_intelligible",
                "m_dim_clarity", "m_dim_calmness")
dim_labels <- c("Mechanicalness", "Expressiveness", "Intelligibility", "Clarity", "Calmness")

dim_models <- map2(dim_vars, dim_files, function(dv, fname) {
  do.call(brm, c(list(
    formula = as.formula(paste(dv, "~ utype_num_z + provenance_z + (1 | participant_id)")),
    data = df, family = gaussian(), prior = priors_gaussian, file = fname), NUTS))
})

check_model <- function(model, name) {
  s    <- summary(model)
  rhat <- max(s$fixed[, "Rhat"], na.rm = TRUE)
  divs <- nuts_params(model) %>% filter(Parameter == "divergent__") %>% pull(Value) %>% sum()
  ess  <- min(s$fixed[, "Bulk_ESS"], na.rm = TRUE)
  message(sprintf("%-20s | Rhat = %.3f | divergences = %d | min ESS = %.0f",
                  name, rhat, divs, ess))
}

walk2(list(m_quality, m_revisions, m_duration, m_confidence, m_vote, m_correct),
      c("quality", "revisions", "duration", "confidence", "vote", "correct"),
      check_model)

draws_vote    <- as_draws_df(m_vote)
draws_correct <- as_draws_df(m_correct)
draws_quality <- as_draws_df(m_quality)
draws_rev     <- as_draws_df(m_revisions)
draws_dur     <- as_draws_df(m_duration)
draws_conf    <- as_draws_df(m_confidence)

a_quality <- draws_quality$b_utype_num_z
a_rev     <- draws_rev$b_utype_num_z
a_dur     <- draws_dur$b_utype_num_z
a_conf    <- draws_conf$b_utype_num_z

indirect_vote <- tibble(
  eval_quality     = a_quality * draws_vote$b_quality_score_composite_z,
  behav_revisions  = a_rev     * draws_vote$b_revision_count,
  behav_duration   = a_dur     * draws_vote$b_log_duration_z,
  behav_confidence = a_conf    * draws_vote$b_rating_confidence_z,
  behav_total      = a_rev * draws_vote$b_revision_count +
                     a_dur * draws_vote$b_log_duration_z +
                     a_conf * draws_vote$b_rating_confidence_z
)

indirect_correct <- tibble(
  eval_quality     = a_quality * draws_correct$b_quality_score_composite_z,
  behav_revisions  = a_rev     * draws_correct$b_revision_count,
  behav_duration   = a_dur     * draws_correct$b_log_duration_z,
  behav_confidence = a_conf    * draws_correct$b_rating_confidence_z,
  behav_total      = a_rev * draws_correct$b_revision_count +
                     a_dur * draws_correct$b_log_duration_z +
                     a_conf * draws_correct$b_rating_confidence_z
)

summarise_draws <- function(df_draws, label) {
  cat("\n===", label, "===\n")
  df_draws %>%
    summarise(across(everything(), list(
      mean  = mean,
      lo95  = \(x) quantile(x, 0.025),
      hi95  = \(x) quantile(x, 0.975),
      p_pos = \(x) mean(x > 0)
    ))) %>%
    pivot_longer(everything(), names_to = c("path", ".value"),
                 names_sep = "_(?=[^_]+$)") %>%
    print(n = Inf)
}

summarise_draws(indirect_vote,    "Indirect effects: utterance type -> vote")
summarise_draws(indirect_correct, "Indirect effects: utterance type -> correct")

cat("\n=== Dimension-level effects ===\n")
dim_summary <- map2_dfr(dim_models, dim_labels, function(mod, lab) {
  s <- as_draws_df(mod)$b_utype_num_z
  tibble(dimension = lab, mean = mean(s),
         lo95 = quantile(s, 0.025), hi95 = quantile(s, 0.975),
         p_pos = mean(s > 0))
})
print(dim_summary)


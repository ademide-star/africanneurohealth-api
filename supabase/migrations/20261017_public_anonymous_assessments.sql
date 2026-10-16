-- Anonymous public app assessments, separate from secondary research records.
-- Apply in Supabase SQL Editor before enabling this feature.
create table if not exists public.neurohealth_public_assessments (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  assessment_type text not null check (assessment_type in (
    'stroke_predictions','alzheimer_predictions','cognitive_symptom_assessments',
    'nutrition_tracker','stress_assessments'
  )),
  country text,
  state_province text,
  nigeria_geopolitical_zone text,
  data_json jsonb not null default '{}'::jsonb,
  model_name text,
  model_version text
);
create index if not exists neurohealth_public_assessments_type_created_idx
  on public.neurohealth_public_assessments (assessment_type, created_at desc);
create index if not exists neurohealth_public_assessments_geo_idx
  on public.neurohealth_public_assessments (country, state_province, assessment_type);
alter table public.neurohealth_public_assessments enable row level security;
-- Deliberately create no anon/authenticated policies. Only the backend service-role key accesses this table.
revoke all on public.neurohealth_public_assessments from anon, authenticated;
grant all on public.neurohealth_public_assessments to service_role;

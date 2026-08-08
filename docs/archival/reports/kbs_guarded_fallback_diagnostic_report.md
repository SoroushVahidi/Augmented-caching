# Guarded fallback diagnostic report

Completed 200-request prefix diagnostic on brightkite and citibike cap128.
- brightkite cap128: guarded is tie than unguarded; miss gap=+0; guard_triggers=0; fallback_time_steps=0; early_return_events=0
- citibike cap128: guarded is tie than unguarded; miss gap=+0; guard_triggers=0; fallback_time_steps=0; early_return_events=0

Interpretation: the fallback/guard mechanism is effectively inactive on this prefix, so it does not currently provide evidence of rescuing the learned policy.
Installed scikit-learn version: 1.8.0
The model loads and runs in the guarded path; the warning seen during unpickling is a version mismatch warning, not a fallback event.

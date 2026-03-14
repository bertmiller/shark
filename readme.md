### SHARK
Scheduler Heuristic Autoresearch with Retrospective Knowledge is an experiment I'm working on.

The basic idea is that a scheduler with knowledge of the future — exact arrival times, output lengths, prefix reuse patterns, etc — can make better decisions than a real scheduler. SHARK attempts to exploit this by having agents optimize what actions the scheduler takes during replays of mooncake-traces, which are traces of real production workloads the Kimi team saw. The output we're trying for is a list of actions the "optimal" scheduler would have taken, which can then act as a labeled dataset for future distilling into heuristics with only information available at scheduling time.

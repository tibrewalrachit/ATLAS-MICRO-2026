#===========================================================================
# atlas.sdc -- timing constraints for a single ATLAS block
#
# Units are set explicitly by run_sta.sh before this file is read, so the
# numbers here are in nanoseconds regardless of the library's own time unit
# (ASAP7 declares ps, sky130 declares ns).
#
# CLK_PERIOD_NS, DRIVE_CELL, DRIVE_PIN and LOAD_FF are set by the caller.
#===========================================================================

create_clock -name clk -period $CLK_PERIOD_NS [get_ports clk]

# A modest 5% uncertainty stands in for the clock-tree skew and jitter that a
# post-CTS run would annotate.  Nothing here has been placed, so this is the
# honest way to keep the report from being optimistic.
set_clock_uncertainty [expr {$CLK_PERIOD_NS * 0.05}] [get_clocks clk]
set_clock_transition  [expr {$CLK_PERIOD_NS * 0.02}] [get_clocks clk]

# Treat the block as sitting inside a larger core: budget 15% of the period
# for arrival at the inputs and 15% for the downstream path at the outputs.
set input_ports  [all_inputs -no_clocks]
set io_budget    [expr {$CLK_PERIOD_NS * 0.15}]

set_input_delay  -clock clk $io_budget $input_ports
set_output_delay -clock clk $io_budget [all_outputs]

set_driving_cell -lib_cell $DRIVE_CELL -pin $DRIVE_PIN $input_ports
set_load $LOAD_FF [all_outputs]

# Reset is asynchronous; it is not a timed path in this analysis.
if {[llength [get_ports -quiet rst_n]] > 0} {
  set_false_path -from [get_ports rst_n]
}

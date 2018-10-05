# SimMETIS
SimMETIS is a data simulator for ELT/METIS, based on SimCADO
It includes a notebook for each METIS mode.

## Bug reports

Please include the output of the following command in all bug reports:

    >>> simmetis.bug_report()

## Notes for the 0.1 release
What is included in the simulation and what isn't?
For the imaging simulation, no detector noise is included. Dark current is irrelevant for all imaging filters and DITs, readout noise isn't completely, but at realistic DITs (filling 70% of the full well capacity) read-out noise never contributes more than about 10% to the total (shot-noise dominated) noise budget. It will be added later.

ELFN can be turned on or off as required. We hope you can use this simulator to show the effect of ELFN on your science application.

The final data product of the imaging simulator is a chop-difference frame. In the simulation you need to give the chopping parameter (chop throw and angle), the simulator will then produce two images and subtract them from each other.

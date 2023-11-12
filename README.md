# molecular_docking
Figures, scripts, and trajectories for my senior thesis project at Dartmouth College. 

I am currently investigating whether molecular docking programs can reproduce relative ligand binding affinities and bound ligand conformations for intrinsically disordered proteins. To do so, I am implementing both AutoDock Vina, a conventional, force-field based docking method, and DiffDock, a new program that using a deep-learning approach to docking. I am docking on molecular dynamics simulations of the disordered region of alpha-synuclein, a protein associated with the Lewy bodies of Parkinson's disease. These molecular dynamics simulations have been clustered using t-distributed stochastic neighbor embedding (t-SNE) methods. My docking protocol 'docks' a ligand of interest on each frame of a protein trajectory. For my analysis, I've randomly selected 1000 frames from each of the 20 t-SNE clusters from the original molecular dynamics simulation to dock on. I've docked on apo simulations of just the protein, as well as other holo simulations where the ligand has simply been removed prior to docking. 

To date, these are the current ligands I have docked: Ligand 41 (Fasudil), Ligand 47, and preliminary runs with Ligand 23. 


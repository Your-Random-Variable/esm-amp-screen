# PyMOL Visualization Script for ESM-AMP-Screen Hits
reinitialize
bg_color white
set ray_shadows, 0
set antialias, 2
load /home/lab202/.gemini/antigravity/scratch/esm_amp_screen/screening_results/target_receptor.pdb, receptor
color gray80, receptor
show cartoon, receptor
set cartoon_transparency, 0.2, receptor

# Hit 1: Magainin_2_frag (Vina: -12.99 kcal/mol, HADDOCK: -95.4, ML-Kd: 236.1 nM, Composite: 15.84)
load /home/lab202/.gemini/antigravity/scratch/esm_amp_screen/screening_results/top_complexes/complex_Magainin_2_frag.pdb, hit_1_Magainin_2_frag
hide everything, hit_1_Magainin_2_frag
show cartoon, hit_1_Magainin_2_frag and chain B
show sticks, hit_1_Magainin_2_frag and chain B and not (name c,n,o)
color cyan, hit_1_Magainin_2_frag and chain B
select pocket_1, (receptor within 4.5 of (hit_1_Magainin_2_frag and chain B))
show sticks, pocket_1 and not (name c,n,o)
color salmon, pocket_1
distance contacts_1, (hit_1_Magainin_2_frag and chain B and (name NZ,NH1,NH2)), (receptor and (name OD1,OD2,OE1,OE2)), 4.0
color yellow, contacts_1

# Hit 2: Indolicidin (Vina: -11.41 kcal/mol, HADDOCK: -77.8, ML-Kd: 2.0 uM, Composite: 14.36)
load /home/lab202/.gemini/antigravity/scratch/esm_amp_screen/screening_results/top_complexes/complex_Indolicidin.pdb, hit_2_Indolicidin
hide everything, hit_2_Indolicidin
show cartoon, hit_2_Indolicidin and chain B
show sticks, hit_2_Indolicidin and chain B and not (name c,n,o)
color magenta, hit_2_Indolicidin and chain B
select pocket_2, (receptor within 4.5 of (hit_2_Indolicidin and chain B))
show sticks, pocket_2 and not (name c,n,o)
color salmon, pocket_2
distance contacts_2, (hit_2_Indolicidin and chain B and (name NZ,NH1,NH2)), (receptor and (name OD1,OD2,OE1,OE2)), 4.0
color yellow, contacts_2

zoom pocket_1, 6.0
set dash_gap, 0.2
set dash_width, 2.5
message 'Loaded ESM-AMP-Screen top hits successfully!'
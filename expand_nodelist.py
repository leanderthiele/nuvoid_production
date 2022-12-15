from sys import argv

slurm_nodelist = argv[1].split(',')

nodelist = []

for slurm_pattern in slurm_nodelist :
    if '[' not in slurm_pattern :
        assert ']' not in slurm_pattern
        nodelist.append(slurm_pattern)
    else :
        assert ']' in slurm_pattern
        root = slurm_pattern[:slurm_pattern.find('[')]
        lo, hi = list(map(int, slurm_pattern[slurm_pattern.find('[')+1:slurm_pattern.find(']')].split('-')))
        for ii in range(lo, hi+1) :
            nodelist.append(f'{root}{ii}')

print(','.join(nodelist))

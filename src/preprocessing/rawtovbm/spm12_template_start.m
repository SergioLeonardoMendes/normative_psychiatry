% List of open inputs
nrun = 1; % enter the number of runs here
jobfile = {'/data/XXX_JOBFILE_XXX'};
jobs = repmat(jobfile, 1, nrun);
inputs = cell(0, nrun);
for crun = 1:nrun
end
spm('defaults', 'PET');
spm_jobman('run', jobs, inputs{:});

function srmr_main(models, dirs)

	files = {'best_checkpoint_23_epoch'}
	rooms = {'REVERB_et_far_room1', 'REVERB_et_far_room2', 'REVERB_et_far_room3', 'REVERB_et_near_room1', 'REVERB_et_near_room2', 'REVERB_et_near_room3'}

	mKinds = {'enhancements'}

	%models = 'ecctn_1'
	%dirs = '/Work19/2019/nabil/ecctn_1`/'

	fid_whole_Name = ['./srmr_results.txt']
	fid_whole =fopen(fid_whole_Name,'w');

	for fmodels = 1 : 1;
	    fprintf(fid_whole, '%s\n', char(mKinds{fmodels}))

	    for flines  = 1 : length(files)
	   
		fileName = ['./matlab/SRMR/MODELS/srmr_ecctn_1.txt'];
		fid=fopen(fileName,'w');

		clnName = ['./matlab/SRMR/MODELS/REVERB_et/REVERB_et_clean_all.txt'];
		lines = importdata(clnName);

		[m, n] = size(lines)

		sum = 0;

		for i = 1 : m;
	 
			cleanSplit = strsplit(char(lines{i}));
			reverbPath = [dirs 'enhancements/best_checkpoint_23_epoch/' char(cleanSplit{1}) '_ch1.wav'];

			% srmrscore = SRMR(cleanSplit{2});
			srmrscore = SRMR(reverbPath );
			sum = sum + srmrscore;
			fprintf(fid, '%s %f\n', cleanSplit{1}, srmrscore);
		  
		end

		fprintf(fid, 'The averange of whole %s is %f\n', char(files{flines}), sum/m)

		fclose(fid);

		fprintf(fid_whole, '%s : %f\n', char(files{flines}), sum/m)

	    end
	end
	fclose(fid_whole)

end
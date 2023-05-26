function pesq_main(models, dirs)

	files = {'best_checkpoint_23_epoch'}
	%models = 'ecctn_1'
	%dirs = '/Work19/2019/nabil/ecctn_1`/'
	rooms = {'REVERB_et_far_room1', 'REVERB_et_far_room2', 'REVERB_et_far_room3', 'REVERB_et_near_room1', 'REVERB_et_near_room2', 'REVERB_et_near_room3'}
	mKinds = {'enhancements'}

	fid_whole_Name = ['./pesq_results.txt']
	fid_whole =fopen(fid_whole_Name,'w');

	for fmodels = 1 : 1;

	    fprintf(fid_whole, '%s\n', char(mKinds{fmodels}));

	    for flines  = 1 : length(files);
	   
		fileName = ['./matlab/PESQ/MODELS/pesq_ecctn_1.txt'];
		fid=fopen(fileName,'w');

		clnName = ['./matlab/PESQ/MODELS/REVERB_et/REVERB_et_clean_all.txt'];
		lines = importdata(clnName);

		[m, n] = size(lines);

		sum = 0;

		for i = 1 : m;
	 
			cleanSplit = strsplit(char(lines{i}));
			revebPath = [dirs 'enhancements/best_checkpoint_23_epoch/' char(cleanSplit{1}) '_ch1.wav'];


			pesqscore = pesq(cleanSplit{2}, revebPath );
			sum = sum + pesqscore;
			fprintf(fid, '%s %f\n', cleanSplit{1}, pesqscore);
		  
		end

		fprintf(fid, 'The averange of whole %s is %f\n', char(files{flines}), sum/m);
		fclose(fid);
		
		fprintf(fid_whole, '%s : %f\n', char(files{flines}), sum/m);

	    end
	end
	fclose(fid_whole)

end
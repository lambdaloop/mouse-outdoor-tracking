addpath('C:\Users\voigtsj\Documents\GitHub\spatial_cognition_analysis\common');
set(0,'DefaultFigureWindowStyle','docked')


clear datapath


c=0;
c=c+1; datapath{c}='V:\outdoor\2025_09_15_mouse_day1\data'; % 1
c=c+1; datapath{c}='V:\outdoor\2025_09_16_mouse_day2\data'; % 2
c=c+1; datapath{c}='V:\outdoor\2025_09_17_mouse_day3\data'; % 3
c=c+1; datapath{c}='V:\outdoor\2025_09_18_mouse_day4\data'; % 4

c=c+1; datapath{c}='V:\outdoor\2025_09_23_mouse_new_day1\data'; % 5
c=c+1; datapath{c}='V:\outdoor\2025_09_24_mouse_new_day2\data'; % 6

c=c+1; datapath{c}='V:\outdoor\2025_09_25_mouse_new_day3\data'; % 7
%c=c+1; datapath{c}='V:\outdoor\2025_06_08_mouse\data'; % 8
c=c+1; datapath{c}='V:\outdoor\2025_10_13_newmouse_d1\data'; % 8

c=c+1; datapath{c}='V:\outdoor\2025_10_20\data'; % 9
c=c+1; datapath{c}='V:\outdoor\2025_10_22\data'; % 10

c=c+1; datapath{c}='V:\outdoor\2025_11_03_day1_x\data'; % 11
c=c+1; datapath{c}='V:\outdoor\2025_11_04_day2\data'; % 12
c=c+1; datapath{c}='V:\outdoor\2025_11_05_day3\data'; % 13
c=c+1; datapath{c}='V:\outdoor\2025_11_06_day4\data'; % 14

c=c+1; datapath{c}='V:\outdoor\2026_04_10_mouse_right\data'; % 15

datapath'

% find(strcmp({d.name},'video_17_2025-09-15T19_22_16.avi'))

%disp('pausing for  a few hrs')
%pause(60*15)

for sess=15:numel(datapath) %
    d=dir(fullfile(datapath{sess},'*.avi'))

    ifplot=1;

    if sess==15 % restart if interrupted
        runvids=[179:600];
    else
        runvids=[1:numel(d)];
    end

    %runvids=(92*11)+12
    
    for vidnum=runvids;%startat:numel(d) 
        %%
        tracks=[];
        sync_pixel=[];
        pixelsizes=[];
        vidname=fullfile(d(vidnum).folder,d(vidnum).name);
        try
            vidObj = VideoReader(vidname)
            nframes=floor((vidObj.Duration*vidObj.FrameRate));%vidObj.NumFrames;

        catch
            warning(sprintf('video %s is corrupted/has 0 bytes, skipping',vidname))
            nframes=0;
        end
        %disp('getting Nframes');
        % nframes=vidObj.NumFrames;



        %%
        if nframes>0
            %% automatically find thermal target?
            if 0
                fprintf('finding thermal target\n')
                if vidnum==1
                    Nf=1200;

                else
                    Nf=5000;

                end
                stack=zeros(512,640,Nf);
                %re-open
                clear vidObj;
                vidObj = VideoReader(vidname);
                for  i=[1:Nf]
                    tmp=readFrame(vidObj);
                    stack(:,:,i)=tmp(:,:,1);
                    if mod(i,100)==0
                        i;
                        clf;imagesc(stack(:,:,i));drawnow;
                    end
                end

                %re-open
                clear vidObj;
                vidObj = VideoReader(vidname);

                %K=kurtosis(stack,[],3);
                %K=std(stack,[],3);
                stack_fft = fft(stack, [], 3);

                indicator_band_im=(mean(abs(stack_fft(:,:,20:50)),3));
                indicator_band_im=( (mean(abs(stack_fft(:,:,20:50)),3))-(mean(abs(stack_fft(:,:,5:15)),3))./4 ); % normalize to avoid grass movement
                imagesc(indicator_band_im); hold on;



                [maxVal, linearIndex] = max(abs(indicator_band_im(:)));
                [track_point_auto(2)  track_point_auto(1) ] = ind2sub(size(indicator_band_im), linearIndex);
                fprintf('found %d %d \n',track_point_auto(1),track_point_auto(2));
                plot(track_point_auto(1),track_point_auto(2),'ro');
                drawnow;
                pause(3)
            end
            %% timestamps for later

            timezone = "America/New_York";
            if 1
                % associated timestamps
                %            d_ts=dir(fullfile(datapath{sess},['timestamps',d(vidnum).name(6:end-5),'?.cvs'])); % allow for seconds to differ in case video starts right at boundary between seconds and names dont fully match
                d_ts=dir(fullfile(datapath{sess},['timestamps',d(vidnum).name(6:end-8),'?_??.cvs'])); % allow for seconds to differ in case video starts right at boundary between seconds and names dont fully match

                assert(numel(d_ts)==1,'got more than one or less than 1 timestamp files for video');
                fid = fopen(fullfile(d_ts(1).folder, d_ts(1).name));
                %  t = textscan(fid, '%d-%d-%dT%d:%d:%f-%d:%d', inf); % 2019-04-13T21:41:13.3566336-04:00,360548
                x = textscan(fid, '%s %u64 %d %d', 'Delimiter', ',');
                times= datetime(x{1}, 'TimeZone', timezone, 'Format', 'yyyy-MM-dd''T''HH:mm:ss.SSSSXXX');
                fclose(fid);
            end

            %% track
            %re-open
            clear vidObj;
            vidObj = VideoReader(vidname);

            %  vidObj.CurrentTime=2955;

            % vidObj.CurrentTime = round(vidObj.Duration*(65.5/100))%;3360 %53*60 +52;
            % vidObj.CurrentTime = 3*(60^2)+6*60 +0;
            bg_ref=zeros(vidObj.Width,vidObj.Height)';
            I=bg_ref;
            tic;
            c=0;

            kal_jumpradius=10;
            kal_xy=[0 0];

            % find cam_id
            tmp=strsplit(d(vidnum).name,'_');
            cam_id =str2num(tmp{2})-9;
            fprintf('cam id: %d\n',cam_id);

            %vidObj.CurrentTime =0;
            while hasFrame(vidObj)
                c=c+1;
                tmp=readFrame(vidObj);

                vidFrame = double(tmp(:,:,1));

                if 1
                    % blank out non-arena areas
                    if cam_id==1
                        track_point=[373 115];
                        trackpoint_blank_x=[-4:4];
                        trackpoint_blank_y=[-2:3];
                        vidFrame(1:1,:)=0;
                                                vidFrame(1:50,400:end)=0;
                                                vidFrame(1:50,1:100)=0;

                        %  if sess<3
                        %      for ix=1:400
                        %          vidFrame(max(1,ceil(300+ix/2)):end,1:ix)=0;
                        %      end
                        %  end
                        %    vidFrame(1:120,500:end)=0;

                    end
                    if cam_id==2
                        vidFrame(1:10,:)=0;
                        vidFrame(1:20,1:50)=0;
                        %vidFrame(1:100,500:end)=0;
                        track_point=[342 151];
                        trackpoint_blank_x=[-5:8];
                        trackpoint_blank_y=[-1:3];
                    end
                    if cam_id==3
                        % vidFrame(1:60,1:120)=0;
                        track_point=[243 46];
                        trackpoint_blank_x=[-4:13];
                        trackpoint_blank_y=[-5:5];

                    end
                    if cam_id==4
                        % vidFrame(1:80,:)=0;
                        track_point=[143 198];
                        trackpoint_blank_x=[-8:5];
                        trackpoint_blank_y=[-4:3];
                    end
                    if cam_id==5
                        track_point=[313 119];
                        trackpoint_blank_x=[-5:5];
                        trackpoint_blank_y=[-2:2];
                        vidFrame(1:50,500:end)=0;


                    end

                    if cam_id==6
                        track_point=[ 360 100  ]; %?
                        trackpoint_blank_x=[-8:3];
                        trackpoint_blank_y=[-2:2];
                        vidFrame(1:70,310:end)=0;
                        vidFrame(1:120,450:end)=0;
                    end

                    if cam_id==7
                        track_point=[ 534 191]; %?
                        trackpoint_blank_x=[-8:3];
                        trackpoint_blank_y=[-2:2];

                    end

                    if cam_id==8
                        track_point=[ 260 144];
                        trackpoint_blank_x=[-5:6];
                        trackpoint_blank_y=[-2:2];
                                                vidFrame(1:50,400:end)=0;

                    end

                    if cam_id==9
                        %vidFrame(1:50,500:end)=0;
                        vidFrame(1:50,1:50)=0;
                        trackpoint_blank_x=[-14:6];
                        trackpoint_blank_y=[-4:5];
                        track_point=[60 440];

                    end
                    if cam_id==10
                        trackpoint_blank_x=[-14:6];
                        trackpoint_blank_y=[-4:5];
                        track_point=[163 118];
                        vidFrame(1:60,1:100)=0;
                        vidFrame(1:20,1:250)=0;
                    end

                    if cam_id==11
                        trackpoint_blank_x=[-14:6];
                        trackpoint_blank_y=[-4:5];
                        track_point=[25 139];

                    end

                    if cam_id==12
                        trackpoint_blank_x=[-14:6];
                        trackpoint_blank_y=[-4:5];
                        track_point=[264 102];

                    end

                end


                %                if norm(track_point-track_point_auto) > 20 % just for safety
                %                    disp('found thermal target too far away/was not defined, defaulting to hard-coded')
                track_point_auto=track_point;
                %               end

                %always use manual
                track_point_auto=track_point;

                curr_dur=toc;
                adapt_rate=0.97;%95;

                if c==1
                    adapt_rate=0;
                end


                mixfactor=0.0; % do we lowpass the images themselves
                I= mixfactor*I + (1-mixfactor)*(vidFrame-bg_ref);

                bg_ref=(1-adapt_rate)*vidFrame + adapt_rate*bg_ref; % slowly adapt background


                tracks(c,:)=[NaN,NaN,NaN];
                sync_pixel(c)=vidFrame(track_point_auto(2),track_point_auto(1)); % get sync pixel from raw video
                tracks(c,3)=sync_pixel(c);

                I(track_point_auto(2)+trackpoint_blank_y,track_point_auto(1)+trackpoint_blank_x)=0; % dont track the sync pixel

                kal_jumpradius=kal_jumpradius+3;


                if true % max(I(:))>10

                    %I_sm=imgaussfilt(I,1,'FilterDomain','auto');
                    I_sm= imboxfilt(I,3);
                    if max(I_sm(:))>20
                        I_sm(1,1)=10;
                        binaryImage=I_sm>15;

                        cc = bwconncomp(binaryImage);
                        pixelSizes = cellfun(@numel, cc.PixelIdxList);

                        minSize = 4;
                        maxSize=1200;

                        validComponents = find( pixelSizes >= minSize & pixelSizes <  maxSize );
                        [~,biggest]=max(pixelSizes);

                        if numel(validComponents)>0
                            R=regionprops(cc);

                            if vecnorm(R(biggest).Centroid-kal_xy) < kal_jumpradius



                                tracks(c,1:2)=R(biggest).Centroid;
                                pixelsizes(c)=R(biggest).Area;

                                kal_xy=R(biggest).Centroid;
                                kal_jumpradius=30;
                            end
                        end;
                    end
                end

               if mod(c,1000)==0
             %   c
            end
                if  mod(c,5000)==0
                    fprintf('sess %d vid %d/%d [%s] frame %d/%d, %.1f%% (%.1f fps) ETA:%.1fhrs\n',sess,vidnum,numel(d),vidname,c,ceil(nframes),100*(c./nframes),c/curr_dur,((curr_dur/c)*(nframes-c))/(60^2));

                    if ifplot
                        clf;
                        subplot(5,1,[1 2]);
                        imagesc(I_sm);
                        hold on;
                        ii=[-200000:0]+c-1; ii=max(ii,1);
                        plot(tracks(ii,1),tracks(ii,2),'w');

                        th = 0:pi/50:2*pi;
                        r=kal_jumpradius;
                        plot(kal_xy(1)+sin(th)*r,kal_xy(2)+cos(th)*r,'r');
                        plot(track_point_auto(1),track_point_auto(2),'ro');
                        daspect([1 1 1]);

                        subplot(5,1,[3 4]);
                        imagesc(vidFrame); hold on;
                        ii=[-1000:0]+c-1; ii=max(ii,1);
                        plot(tracks(ii,1),tracks(ii,2),'w');
                        plot(track_point_auto(1),track_point_auto(2),'ro');
                        daspect([1 1 1]);

                        subplot(5,1,[5]);
                        plot(sync_pixel);

                        drawnow;
                    end


                end
            end

            tracks(vidObj.NumFrames,1)=0;
            disp('done with video');
            outfolder=strrep(d(vidnum).folder,'\outdoor\','\outdoor_analysis\');
            mkdir(outfolder);
            outname= fullfile(outfolder,d(vidnum).name);
            outname(end-2:end)='mat';
            save(outname,'tracks','pixelsizes','times');
        end
    end
end
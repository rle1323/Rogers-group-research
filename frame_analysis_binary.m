DataFile = 'Config0_Time.txt';   %csv file contains strain in 1st column, stress in 2nd column, and time in 3rd column
VideoFile = '64_data30.txt';      %video must be in mp4 format

%% import data
% video
cd 'om0'
video= importdata(VideoFile);
cd '..'
nn = size(video, 1);
mm = sqrt(size(video, 2));
% data
%data = importdata(DataFile);

%% Make Video
% Read movie and data frames

%name = 'test';
%writerObj = VideoWriter(name);
% writerObj.FrameRate = rate;
%open(writerObj); 
for ii = 1:nn
%     for jj = 1:a
%         if zeroedrunData(jj,3) >= time
%             pos = jj;
%             break
%         end
%     end

    fig = figure(1);clf
    set(gcf, 'Position', [0, 220, 1280, 500]);
    
    %%%%%%%%%%%%%%%%%%% PLOT THE SS DATA %%%%%%%%%%%%%%%%%%%%
    %runData = data;
    %ax1 = subplot(1,2,1);
    %hold on
    %h1 = plot(runData(1:ii,3), runData(1:ii,2),'k.-');
    %h2 = plot(runData(ii:end,3), runData(ii:end,2),'c.-');
    %h3 = plot(runData(ii,3), runData(ii,2),'ro');
    %hold off
    %set(h1,'LineWidth',2)
    %set(h2,'color',[0.5 0.5 0.5])
    %set(h3,'MarkerSize',16,'MarkerFaceColor','r')
    
    %tName = ['Notched: 3um Unit cell, Wall thickness = 10nm (Video Speed: ' num2str(speed) 'x)'];
    %xlabel('True Strain', 'FontSize', 20)
    %%ylabel('True Stress (GPa)', 'FontSize', 20)
    %set(figure(1), 'color', 'white'); % sets the color to white
    %set(gca,'FontSize', 18);
    %title(tName,'FontWeight', 'bold', 'FontSize', 18)
    
    %%%%%%%%%%%%%%%%%% PLOT 2D STRAIN FIELD %%%%%%%%%%%%%%%%%%%%
    ax2 = subplot(1,2,2);
    rawVideo = video(ii,:);
    Vi = ones(mm, mm);
    %Vi(rawVideo>median(rawVideo))=0;
    if ii>1
        %prev_video = video(ii-1,:);
        %Vi(rawVideo>prev_video)=0;
        Vi(rawVideo>median(rawVideo))=0;
    else
        continue
    end
    
    cd 64_images/30;
    imwrite(reshape(Vi, mm, mm),strcat('frame_', int2str(ii), '.png'));
    %stress = stress + .5;
    cd ../..;
    
    % set frame
%    frame = getframe(gcf);
%    writeVideo(writerObj, frame);
end

close(writerObj);
disp('Video frames have been created.')
close all

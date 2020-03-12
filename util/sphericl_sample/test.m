% Uniformly distribute 200 charged particles across unit sphere
[V,Tri,~,Ue]=ParticleSampleSphere('N',90);

% Visualize optimization progress
figure('color','w')
plot(log10(1:numel(Ue)),Ue,'.-')
set(get(gca,'Title'),'String','Optimization Progress','FontSize',40)
set(gca,'FontSize',20,'XColor','k','YColor','k')
xlabel('log_{10}(Iteration #)','FontSize',30,'Color','k')
ylabel('Reisz s-Energy','FontSize',30,'Color','k')

% Visualize mesh based on computed configuration of particles
figure('color','w')
subplot(1,2,1)
fv=struct('faces',Tri,'vertices',V);
h=patch(fv);
set(h,'EdgeColor','b','FaceColor','w')
axis equal
hold on
plot3(V(:,1),V(:,2),V(:,3),'.k','MarkerSize',15)
set(gca,'XLim',[-1.1 1.1],'YLim',[-1.1 1.1],'ZLim',[-1.1 1.1])
view(3)
grid off
set(get(gca,'Title'),'String','N=90 (base mesh)','FontSize',30)

% Subdivide base mesh twice to obtain a spherical mesh of higher complexity
fv_new=SubdivideSphericalMesh(fv,2);
subplot(1,2,2)
h=patch(fv_new);
set(h,'EdgeColor','b','FaceColor','w')
axis equal
hold on
plot3(V(:,1),V(:,2),V(:,3),'.k','MarkerSize',15)
set(gca,'XLim',[-1.1 1.1],'YLim',[-1.1 1.1],'ZLim',[-1.1 1.1])
view(3)
grid off
set(get(gca,'Title'),'String','N=3170 (after 2 subdivisions)','FontSize',30);
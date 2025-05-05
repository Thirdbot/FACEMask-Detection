import AppContainer from "../containers/AppContainer";
import Sidebar from "../ui/Sidebar";
import PageContent from "../containers/PageContent";
import Title from "../ui/Title";

const ProjectDescription = () => {
  return (
    <AppContainer>
      <Sidebar />
      <PageContent>
        <Title text="รายละเอียดโปรเจค" />
        <p>
          Lorem ipsum dolor sit amet consectetur adipisicing elit. Dolorum ab
          mollitia reiciendis sequi numquam distinctio, veritatis modi dolorem
          ipsum sed? Optio asperiores quia fugit odio perferendis culpa rerum
          pariatur dolor?
        </p>
      </PageContent>
    </AppContainer>
  );
};

export default ProjectDescription;

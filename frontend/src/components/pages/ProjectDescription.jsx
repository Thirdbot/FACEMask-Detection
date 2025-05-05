import AppContainer from "../containers/AppContainer";
import Sidebar from "../ui/Sidebar";
import PageContent from "../containers/PageContent"
import Title from "../ui/Title"

const ProjectDescription = () => {
  return (
    <AppContainer>
      <Sidebar/>
      <PageContent>
        <Title text="รายละเอียดโปรเจค" />
      </PageContent>
    </AppContainer>
  )
}

export default ProjectDescription
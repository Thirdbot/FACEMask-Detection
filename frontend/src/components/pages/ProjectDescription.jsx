import AppContainer from "../containers/AppContainer";
import Sidebar from "../ui/Sidebar";
import PageContent from "../containers/PageContent"

const ProjectDescription = () => {
  return (
    <AppContainer>
      <Sidebar/>
      <PageContent>
        <h1>หน้าแสดงรายละเอียดโปรเจค</h1>
      </PageContent>
    </AppContainer>
  )
}

export default ProjectDescription
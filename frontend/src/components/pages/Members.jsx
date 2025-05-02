import AppContainer from "../containers/AppContainer";
import Sidebar from "../ui/Sidebar";
import PageContent from "../containers/PageContent"

const Members = () => {
  return (
    <AppContainer>
      <Sidebar/>
      <PageContent>
        <h1>สมาชิกในกลุ่ม</h1>
      </PageContent>
    </AppContainer>
  )
}

export default Members
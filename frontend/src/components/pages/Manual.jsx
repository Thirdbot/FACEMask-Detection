import AppContainer from "../containers/AppContainer";
import Sidebar from "../ui/Sidebar";
import PageContent from "../containers/PageContent";

const Manual = () => {
  return (
    <AppContainer>
      <Sidebar />
      <PageContent>
        <h1>หน้าสอนการใช้งาน</h1>
      </PageContent>
    </AppContainer>
  );
};

export default Manual;

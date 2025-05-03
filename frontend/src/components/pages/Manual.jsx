import AppContainer from "../containers/AppContainer";
import Sidebar from "../ui/Sidebar";
import PageContent from "../containers/PageContent";
import Title from "../ui/Title";

const Manual = () => {
  return (
    <AppContainer>
      <Sidebar />
      <PageContent>
        <Title text="สอนการใช้งาน"/>
      </PageContent>
    </AppContainer>
  );
};

export default Manual;

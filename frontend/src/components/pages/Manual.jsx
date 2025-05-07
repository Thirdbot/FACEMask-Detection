import AppContainer from "../containers/AppContainer";
import Sidebar from "../ui/Sidebar";
import PageContent from "../containers/PageContent";
import Title from "../ui/Title";

const Manual = () => {
  return (
    <AppContainer>
      <Sidebar />
      <PageContent>
        <Title text="วิธีการใช้งาน"/>
        <p>Lorem ipsum dolor sit amet consectetur adipisicing elit. Culpa eum nemo veritatis in quos corporis vitae a, quidem obcaecati, aliquid sit nihil. Assumenda quia sed sequi fugiat alias incidunt illo.</p>
      </PageContent>
    </AppContainer>
  );
};

export default Manual;

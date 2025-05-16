const PageContent = ({ children, className }) => {
  return (
    <div
      className={`px-10 pt-14 w-full ${className ? className : "flex flex-col items-center justify-start"}`}
    >
      {children}
    </div>
  );
};

export default PageContent;

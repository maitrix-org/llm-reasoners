(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects h j f l)
(:init 
(handempty)
(ontable h)
(ontable j)
(ontable f)
(ontable l)
(clear h)
(clear j)
(clear f)
(clear l)
)
(:goal
(and
(on h j)
(on j f)
(on f l)
)))
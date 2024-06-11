(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects e a j f)
(:init 
(handempty)
(ontable e)
(ontable a)
(ontable j)
(ontable f)
(clear e)
(clear a)
(clear j)
(clear f)
)
(:goal
(and
(on e a)
(on a j)
(on j f)
)))
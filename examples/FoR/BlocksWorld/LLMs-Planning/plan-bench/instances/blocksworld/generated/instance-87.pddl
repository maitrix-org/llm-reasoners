(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects h e b l)
(:init 
(handempty)
(ontable h)
(ontable e)
(ontable b)
(ontable l)
(clear h)
(clear e)
(clear b)
(clear l)
)
(:goal
(and
(on h e)
(on e b)
(on b l)
)))
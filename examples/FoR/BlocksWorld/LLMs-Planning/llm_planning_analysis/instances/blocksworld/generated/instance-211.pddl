(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects f e b a)
(:init 
(handempty)
(ontable f)
(ontable e)
(ontable b)
(ontable a)
(clear f)
(clear e)
(clear b)
(clear a)
)
(:goal
(and
(on f e)
(on e b)
(on b a)
)))
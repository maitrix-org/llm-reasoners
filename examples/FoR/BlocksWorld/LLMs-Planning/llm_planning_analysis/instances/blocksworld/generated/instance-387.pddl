(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects b e f)
(:init 
(handempty)
(ontable b)
(ontable e)
(ontable f)
(clear b)
(clear e)
(clear f)
)
(:goal
(and
(on b e)
(on e f)
)))
(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects a e h b)
(:init 
(handempty)
(ontable a)
(ontable e)
(ontable h)
(ontable b)
(clear a)
(clear e)
(clear h)
(clear b)
)
(:goal
(and
(on a e)
(on e h)
(on h b)
)))
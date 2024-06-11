(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects h b f)
(:init 
(handempty)
(ontable h)
(ontable b)
(ontable f)
(clear h)
(clear b)
(clear f)
)
(:goal
(and
(on h b)
(on b f)
)))